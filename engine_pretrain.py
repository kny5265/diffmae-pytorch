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
import numpy as np
import matplotlib.pyplot as plt
from typing import Iterable

import torch

import util.misc as misc
import util.lr_sched as lr_sched
from util.custom import calc_for_diffmae

def visualize(args, model, samples, visible_input, pred, ids_restore, mask, iter, data_iter_step, mode='train'):
    if data_iter_step % 20 != 0: # only plot every 20 iteration steps
        return
    
    if args.multi_gpu:
        model = model.module
  
    pred = torch.cat([visible_input, pred], dim=1)
    pred = torch.gather(pred, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, pred.shape[2])) # to unshuffle
    pred = model.unpatchify(pred)

    # the first sample of a mini-batch, the first feature
    vis_target = np.array(samples[0, :, 0].cpu().detach())
    vis_pred = np.array(pred[0, :, 0].cpu().detach())

    t = np.arange(int(samples.shape[1]))

    fig, axs = plt.subplots(2,1)
    fig.tight_layout(pad=2)

    axs[0].plot(t,vis_target)
    axs[0].set_title('ground truth sequence')

    axs[1].plot(t,vis_pred)
    axs[1].set_title('predicted sequence')

    """ gray area generation """

    mask_ = torch.ones(samples.shape[0], samples.shape[1]).to(args.device)

    for b in range(samples.shape[0]):
        for n in range(args.input_length):
            index = n // args.patch_size
            mask_[b, n] *= mask[b, index]

    start_idx, final_idx = 0, 0
    start_prev, final_prev = 99999, 99999
    for i in range(len(mask_[0, :])-1):
        if i == 0 or (mask_[0, i] == 1.0 and mask_[0, i-1] == 0.0):
            start_idx = i
        elif mask_[0, i] == 1.0 and mask_[0, i+1] == 0.0:
            final_idx = i
        elif i == len(mask_[0, :])-2 and mask_[0, i] == 1.0:
            final_idx = i
        
        if start_idx < final_idx and (start_prev != start_idx and final_prev != final_idx):

            plt.axvspan(start_idx, final_idx, facecolor='gray', alpha=0.2)

            start_prev = start_idx 
            final_prev = final_idx

    """ done """

    folder = '{}/{}_plot'.format(args.savedir, mode)
    os.makedirs(folder, exist_ok=True)

    if mode == 'train' or 'val':
        plt.savefig('{}/comparison_{}epoch_{}batch.png'.format(folder, iter, data_iter_step))
    else:
        plt.savefig('{}/comparison_{}batch.png'.format(folder, data_iter_step))
    plt.cla()   # clear the current axes
    plt.clf()   # clear the current figure
    plt.close() # closes the current figure

def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    epoch: int, loss_scaler,
                    log_writer=None,
                    args=None, iter=0):
    
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 50

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
            pred, visible_token, ids_masked, ids_restore, mask = model(samples)
            loss = calc_for_diffmae(args, model, samples, pred, ids_restore, ids_masked)

        visualize(args, model, samples, visible_token, pred, ids_restore, mask, epoch, data_iter_step, mode='train')

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
    # metric_logger.synchronize_between_processes()
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
            pred, visible_token, ids_masked, ids_restore, mask = model(samples)
            loss = calc_for_diffmae(args, model, samples, pred, ids_restore, ids_masked)

        visualize(args, model, samples, visible_token, pred, ids_restore, mask, epoch, data_iter_step, mode='test')

        loss_value = loss.item()

        loss /= accum_iter

        metric_logger.update(loss=loss_value)

    # gather the stats from all processes
    # metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}