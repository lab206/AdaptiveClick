from fvcore.common.config import CfgNode

from isegm.utils.exp_imports.default import *
from isegm.engine.adaptiveclick_trainer import AdaptiveClickTrainer
from isegm.model.is_adaptiveclick_model import AdaptiveClickModel

MODEL_NAME = 'adaptiveclick_base448_cocolvis'
MODEL_CONFIG = 'configs/adaptiveclick_plainvit_base448.yaml'


def main(cfg):
    model, model_cfg = init_model(cfg)
    train(model, cfg, model_cfg)


def init_model(cfg):
    # fetch config for model
    model_cfg = CfgNode.load_yaml_with_base(MODEL_CONFIG, allow_unsafe=True)
    # merge args to model_config
    for k, v in cfg.__dict__.items():
        model_cfg[k] = v

    model_cfg = edict(d=model_cfg)
    model_cfg.crop_size = (448, 448)
    model_cfg.num_max_points = 24

    model = AdaptiveClickModel(model_cfg, with_prev_mask=True, use_disks=True, norm_radius=5)
    model.backbone.init_weights_from_pretrained(model_cfg.MODEL.IMAGENET_PRETRAINED_MODELS)
    model.to(cfg.device)
    return model, model_cfg


def train(model, cfg, model_cfg):
    cfg.batch_size = 32 if cfg.batch_size < 1 else cfg.batch_size
    cfg.val_batch_size = cfg.batch_size
    crop_size = model_cfg.crop_size

    loss_cfg = edict()
    loss_cfg.instance_loss_weight = 1.0

    train_augmentator = Compose([
        UniformRandomResize(scale_range=(0.75, 1.40)),
        HorizontalFlip(),
        PadIfNeeded(min_height=crop_size[0], min_width=crop_size[1], border_mode=0),
        RandomCrop(*crop_size),
        RandomBrightnessContrast(brightness_limit=(-0.25, 0.25), contrast_limit=(-0.15, 0.4), p=0.75),
        RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.75)
    ], p=1.0)

    val_augmentator = Compose([
        PadIfNeeded(min_height=crop_size[0], min_width=crop_size[1], border_mode=0),
        RandomCrop(*crop_size)
    ], p=1.0)

    points_sampler = MultiPointSampler(model_cfg.num_max_points, prob_gamma=0.80,
                                       merge_objects_prob=0.15,
                                       max_num_merged_objects=2)

    trainset = CocoLvisDataset(
        cfg.LVIS_v1_PATH,
        split='train',
        augmentator=train_augmentator,
        min_object_area=1000,
        keep_background_prob=0.05,
        points_sampler=points_sampler,
        epoch_len=30000,
        stuff_prob=0.30
    )

    valset = CocoLvisDataset(
        cfg.LVIS_v1_PATH,
        split='val',
        augmentator=val_augmentator,
        min_object_area=1000,
        points_sampler=points_sampler,
        epoch_len=2000
    )

    optimizer_params = {
        'lr': 5e-5, 'betas': (0.9, 0.999), 'eps': 1e-8
    }

    lr_scheduler = partial(
        torch.optim.lr_scheduler.MultiStepLR,
        milestones=[40, 60], gamma=0.1
    )

    trainer = AdaptiveClickTrainer(
        model=model, cfg=cfg, model_cfg=model_cfg, loss_cfg=loss_cfg,
        trainset=trainset, valset=valset,
        optimizer='adam',
        optimizer_params=optimizer_params,
        lr_scheduler=lr_scheduler,
        checkpoint_interval=[(0, 30), (50, 1)],
        image_dump_interval=100,
        metrics=[AdaptiveIoU(from_logits=False)],
        max_interactive_points=model_cfg.num_max_points,
        max_num_next_clicks=3
    )
    trainer.run(num_epochs=60, validation=False)
