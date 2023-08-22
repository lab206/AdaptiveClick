import math

import torch
import torch.nn as nn
from torch.nn import functional as F
from isegm.model.is_model import ISModel
from isegm.model.is_plainvit_model import SimpleFPN
from isegm.model.modeling.mask2former_helper.mask2former_transformer_decoder import MultiScaleMaskedTransformerDecoder
from isegm.model.modeling.mask2former_helper.msdeformattn import MSDeformAttnPixelDecoder
from isegm.utils.serialization import serialize
from isegm.model.criterion import SetCriterion
from isegm.model.matcher import HungarianMatcher
from isegm.model.modeling.models_vit import VisionTransformer, PatchEmbed

from addict import Dict


class PixelLevelMultiScaleDecoder(MSDeformAttnPixelDecoder):
    def __init__(self, cfg, input_shape):
        super().__init__(
            input_shape,
            transformer_dropout=cfg.MODEL.MASK_FORMER.DROPOUT,
            transformer_nheads=cfg.MODEL.MASK_FORMER.NHEADS,
            transformer_dim_feedforward=cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD // 2,
            transformer_enc_layers=cfg.MODEL.SEM_SEG_HEAD.TRANSFORMER_ENC_LAYERS,
            conv_dim=cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM,
            mask_dim=cfg.MODEL.SEM_SEG_HEAD.MASK_DIM,
            transformer_in_features=cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_IN_FEATURES,
            common_stride=cfg.MODEL.SEM_SEG_HEAD.COMMON_STRIDE
        )
        self.in_features = cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES
        self.neck = SimpleFPN(
            in_dim=cfg.MODEL.SIMPLE_FEATURE_PYRAMID.IN_DIM,
            out_dims=cfg.MODEL.SIMPLE_FEATURE_PYRAMID.OUT_DIMS
        )

    def forward(self, features, grid_size):
        # extract 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
        B, N, C = features.shape
        features = features.transpose(-1, -2).view(B, C, grid_size[0], grid_size[1])
        multi_scale_features = {k: v for k, v in zip(self.in_features, self.neck(features))}
        mask_features, transformer_encoder_features, multi_scale_features = self.forward_features(multi_scale_features)
        return mask_features, transformer_encoder_features, multi_scale_features


class ClickAwareMaskAdaptiveDecoder(MultiScaleMaskedTransformerDecoder):
    def __init__(self, cfg):
        super().__init__(
            in_channels=cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM,
            num_classes=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
            mask_classification=True,
            hidden_dim=cfg.MODEL.MASK_FORMER.HIDDEN_DIM,
            num_queries=cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES,
            nheads=cfg.MODEL.MASK_FORMER.NHEADS,
            dim_feedforward=cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD,
            dec_layers=cfg.MODEL.MASK_FORMER.DEC_LAYERS - 1,
            pre_norm=cfg.MODEL.MASK_FORMER.PRE_NORM,
            mask_dim=cfg.MODEL.SEM_SEG_HEAD.MASK_DIM,
            enforce_input_project=False
        )

    def forward_prediction_heads(self, output, mask_features, attn_mask_target_size, mask=None):
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)
        outputs_class = self.class_embed(decoder_output)
        mask_embed = self.mask_embed(decoder_output)
        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)

        # NOTE: prediction is of higher-resolution
        # [B, Q, H, W] -> [B, Q, H*W] -> [B, h, Q, H*W] -> [B*h, Q, HW]
        attn_mask = F.interpolate(outputs_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False)
        # must use bool type If a BoolTensor is provided, positions with ``True`` are not allowed to attend while
        # ``False`` values will be unchanged.
        attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1)
                     .flatten(0, 1) < 0.5).bool()
        attn_mask = attn_mask.detach()

        if mask is not None:
            # tensors with 0 are not applied clicks while 1 are applied clicks
            mask = torch.einsum("bq,bhw->bqhw", outputs_class.argmax(-1), mask)
            num_pooling_layer = int(math.log(mask.shape[-1] / attn_mask_target_size[-1], 2))
            pooling_layer = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            for _ in range(num_pooling_layer):
                mask = pooling_layer(mask)
            mask = F.interpolate(mask, size=attn_mask_target_size, mode="bilinear", align_corners=False)
            click_aware_mask = (mask.flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0, 1) < 0.5).bool()
            attn_mask = attn_mask & click_aware_mask

        return outputs_class, outputs_mask, attn_mask


class AdaptiveClickModel(ISModel):
    @serialize
    # in order to make serialize, cfg should be set default args
    def __init__(self, cfg, with_prev_mask=True, **kwargs):
        super().__init__(with_prev_mask=with_prev_mask, **kwargs)
        self.num_queries = cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES
        self.num_classes = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
        self.device = cfg.device
        self.random_split = cfg.random_split

        self.patch_embed_coords = PatchEmbed(
            img_size=cfg.crop_size,
            patch_size=cfg.MODEL.BACKBONE.PATCH_SIZE,
            in_chans=3 if with_prev_mask else 2,
            embed_dim=cfg.MODEL.BACKBONE.EMBED_DIM,
        )
        self.backbone = VisionTransformer(
            img_size=cfg.crop_size,
            patch_size=cfg.MODEL.BACKBONE.PATCH_SIZE,
            in_chans=cfg.MODEL.BACKBONE.IN_CHANS,
            embed_dim=cfg.MODEL.BACKBONE.EMBED_DIM,
            depth=cfg.MODEL.BACKBONE.DEPTH,
            num_heads=cfg.MODEL.BACKBONE.NUM_HEADS,
            mlp_ratio=cfg.MODEL.BACKBONE.MLP_RATIO,
            qkv_bias=cfg.MODEL.BACKBONE.QKV_BIAS
        )

        self.backbone_feature_shape = dict()
        for i, channel in enumerate(cfg.MODEL.SIMPLE_FEATURE_PYRAMID.OUT_DIMS):
            self.backbone_feature_shape[f'res{i + 2}'] = Dict({'channel': channel, 'stride': 2 ** (i + 2)})

        self.pixel_decoder = PixelLevelMultiScaleDecoder(cfg, self.backbone_feature_shape)
        self.click_aware_decoder = ClickAwareMaskAdaptiveDecoder(cfg)

        self._training_init(cfg)

    def forward(self, image, points):
        image, prev_mask = self.prepare_input(image)
        coord_features = self.get_coord_features(image, prev_mask, points)
        click_aware_features = coord_features[:, 1]
        coord_features = self.patch_embed_coords(coord_features)
        backbone_features = self.backbone.forward_backbone(image, coord_features, shuffle=self.random_split)

        mask_features, transformer_encoder_features, multi_scale_features = self.pixel_decoder(
            backbone_features, self.backbone.patch_embed.grid_size
        )
        predictions = self.click_aware_decoder(multi_scale_features, mask_features, click_aware_features)

        # instances is the mask, predictions has 'pred_logits' and 'pred_mask'
        return {'instances': self.segment_inference(predictions, image.size()), 'predictions': predictions}

    def _training_init(self, cfg):
        # Loss parameters:
        deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT

        # loss weights
        class_weight = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT
        dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
        mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT

        matcher = HungarianMatcher(
            cost_class=class_weight,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
        )

        weight_dict = {"loss_ce": class_weight, "loss_mask": mask_weight, "loss_dice": dice_weight}
        if deep_supervision:
            dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        losses = ["labels", "masks"]
        self.criterion = SetCriterion(
            self.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
            device=self.device
        )

    def train(self, mode=True):
        super().train()
        self.criterion.train()
        return self

    def zero_grad(self, set_to_none=False):
        super().zero_grad()
        self.criterion.zero_grad()

    def segment_inference(self, predictions, image_size):
        """
            just use mask_cls[1] for our target
            we remove the sigmoid method
        """
        mask_cls = predictions["pred_logits"]
        mask_pred = predictions["pred_masks"]

        mask_pred = F.interpolate(
            mask_pred,
            size=(image_size[-1], image_size[-2]),
            mode="bilinear",
            align_corners=True,
        )

        mask_cls = mask_cls.sigmoid()[..., 1:]
        mask = torch.einsum("bqc,bqhw->bchw", mask_cls, mask_pred)
        return mask

    def loss_inference(self, outputs, targets):
        losses = self.criterion(outputs, targets)
        loss_ce = 0.0
        loss_dice = 0.0
        loss_mask = 0.0
        for k in list(losses.keys()):
            if k in self.criterion.weight_dict:
                losses[k] *= self.criterion.weight_dict[k]
                if '_ce' in k:
                    loss_ce += losses[k]
                elif '_dice' in k:
                    loss_dice += losses[k]
                else:
                    loss_mask += losses[k]
            else:
                # remove this loss if not specified in `weight_dict`
                losses.pop(k)
        loss = 0.3 * loss_ce + 0.3 * loss_dice + 0.4 * loss_mask
        return loss
