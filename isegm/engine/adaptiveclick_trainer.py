import random
import torch
from torch.nn import functional as F
import numpy as np
from isegm.engine.trainer import ISTrainer, get_next_points
from isegm.utils.misc import save_checkpoint
from isegm.utils.log import logger


class AdaptiveClickTrainer(ISTrainer):
    def __init__(self, **kwargs):
        """
            Overwrite the ISTrainer for AdaptiveClick model
        """
        super().__init__(**kwargs)

    def run(self, num_epochs, start_epoch=None, validation=True):
        if start_epoch is None:
            start_epoch = self.cfg.start_epoch

        logger.info(f'Starting Epoch: {start_epoch}')
        logger.info(f'Total Epochs: {num_epochs}')

        for epoch in range(start_epoch, num_epochs):
            self.training(epoch)
            if validation:
                self.validation(epoch)

    def batch_forward(self, batch_data, validation=False):
        metrics = self.val_metrics if validation else self.train_metrics
        losses_logging = dict()

        with torch.set_grad_enabled(not validation):
            batch_data = {k: v.to(self.device) for k, v in batch_data.items()}
            image, gt_mask, points = batch_data['images'], batch_data['instances'], batch_data['points']
            orig_image, orig_gt_mask, orig_points = image.clone(), gt_mask.clone(), points.clone()

            prev_output = torch.zeros_like(image, dtype=torch.float32)[:, :1, :, :]

            last_click_indx = None

            with torch.no_grad():
                num_iters = random.randint(0, self.max_num_next_clicks)

                for click_indx in range(num_iters):
                    last_click_indx = click_indx

                    if not validation:
                        self.net.eval()

                    if self.click_models is None or click_indx >= len(self.click_models):
                        eval_model = self.net
                    else:
                        eval_model = self.click_models[click_indx]

                    net_input = torch.cat((image, prev_output), dim=1) if self.net.with_prev_mask else image
                    # get previous output from model (binary mask type)
                    prev_output = torch.sigmoid(eval_model(net_input, points)['instances'])

                    points = get_next_points(prev_output, orig_gt_mask, points, click_indx + 1)

                    if not validation:
                        self.net.train()

                if self.net.with_prev_mask and self.prev_mask_drop_prob > 0 and last_click_indx is not None:
                    zero_mask = np.random.random(size=prev_output.size(0)) < self.prev_mask_drop_prob
                    prev_output[zero_mask] = torch.zeros_like(prev_output[zero_mask])

            batch_data['points'] = points

            net_input = torch.cat((image, prev_output), dim=1) if self.net.with_prev_mask else image
            # (query type)
            output = self.net(net_input, points)

            loss = 0.0
            mask_target = batch_data['instances']
            down_sample_shape = output['predictions']['pred_masks'].shape
            mask_target = F.interpolate(
                mask_target,
                size=(down_sample_shape[-1], down_sample_shape[-2]),
                mode="bilinear",
                align_corners=False,
            )
            mask_target = torch.squeeze(mask_target, 1)
            loss = self.add_loss('instance_loss', loss, losses_logging, validation,
                                 lambda: (output['predictions'], mask_target))

            if self.is_master:
                with torch.no_grad():
                    for m in metrics:
                        m.update(*(output.get(x) for x in m.pred_outputs), *(batch_data[x] for x in m.gt_outputs))

            self.net.zero_grad()

        return loss, losses_logging, batch_data, output

    def add_loss(self, loss_name, total_loss, losses_logging, validation, lambda_loss_inputs):
        loss_cfg = self.loss_cfg if not validation else self.val_loss_cfg
        loss_weight = loss_cfg.get(loss_name + '_weight', 0.0)
        if loss_weight > 0.0:
            loss_criterion = self.net.loss_inference
            loss = loss_criterion(*lambda_loss_inputs())
            losses_logging[loss_name] = loss
            loss = loss_weight * loss
            total_loss = total_loss + loss

        return total_loss
