# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import os
import math
import sys
import imageio
import numpy as np
import matplotlib.pyplot as plt
from typing import Iterable
from PIL import Image

import torch
from torchvision.transforms.functional import to_pil_image

import util.misc as misc
import util.lr_sched as lr_sched
from util.loss import calc_for_diffmae

def concat_images_horizontally(image_list):
    widths, heights = zip(*(img.size for img in image_list))
    
    total_width = sum(widths)
    max_height = max(heights)
    
    new_img = Image.new('RGB', (total_width, max_height))
    
    x_offset = 0
    for img in image_list:
        new_img.paste(img, (x_offset, 0))
        x_offset += img.width
    
    return new_img

def denormalize(tensor):
    tensor = tensor * 0.5 + 0.5
    return tensor

def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    epoch: int, loss_scaler,
                    log_writer=None,
                    args=None, iter=0):
    
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(args.device, non_blocking=True)

        with torch.cuda.amp.autocast():
            pred, ids_restore, mask, ids_masked, _ = model(samples)
            loss = calc_for_diffmae(args, model, samples, pred, ids_restore, ids_masked)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def evaluate(model: torch.nn.Module,
                    data_loader: Iterable,
                    epoch='',
                    log_writer=None,
                    args=None):
    
    model.eval()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test: '
    print_freq = 20

    accum_iter = args.accum_iter

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        samples = samples.to(args.device, non_blocking=True)

        with torch.cuda.amp.autocast():
            pred, ids_restore, mask, ids_masked, ids_keep = model(samples)
            loss = calc_for_diffmae(args, model, samples, pred, ids_restore, ids_masked)

        loss_value = loss.item()

        loss /= accum_iter

        metric_logger.update(loss=loss_value)

        if args.sampling and data_iter_step % 100 == 0:
            if args.multi_gpu:
                model_ = model.module
            else:
                model_ = model

            for n in range(pred.size()[0]):
                if n % 100 == 0:
                    sampled_token = model_.diffusion.sample(pred[n].unsqueeze(0))
                    sampled_token = sampled_token.squeeze()
                    visible_tokens = model_.patchify(samples)

                    visible_tokens = torch.gather(visible_tokens, dim=1, index=ids_keep[:, :, None].expand(-1, -1, visible_tokens.shape[2]))
                    img = torch.cat([visible_tokens[n], sampled_token], dim=0)
                    img = torch.gather(img, dim=0, index=ids_restore[n].unsqueeze(-1).repeat(1, img.shape[1])) # to unshuffle
                    img = model_.unpatchify(img.unsqueeze(0))
                    
                    img = denormalize(img)
                    samples = denormalize(samples)
                
                    img_array = img.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()  # (n_channels, height, width) -> (height, width, n_channels)
                    org_array = samples[n].squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
                    # org_img = to_pil_image(samples[n] * 255)
                    org_img = Image.fromarray((org_array * 255).astype(np.uint8))
                    img = Image.fromarray((img_array * 255).astype(np.uint8))

                    images = [org_img, img]
                    concatenated_image_horizontal = concat_images_horizontally(images)
                    concatenated_image_horizontal.save('{}/sample/output_{}_{}.png'.format(args.savedir, data_iter_step, n))

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
