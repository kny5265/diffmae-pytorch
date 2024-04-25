# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import argparse
import datetime
import json
import numpy as np
import os
import glob
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import timm

assert timm.__version__ == "0.3.2" # version check
import timm.optim.optim_factory as optim_factory
from timm.models.layers import trunc_normal_
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

import util.lr_decay as lrd
import util.misc as misc
from util.datasets import build_dataset
from util.pos_embed import interpolate_pos_embed
from util.misc import NativeScalerWithGradNormCount as NativeScaler

import option
from data import create_dataset
from model import diffusion, diffmae_finetune

from engine_finetune import train_one_epoch, evaluate

def main(args):

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    if misc.get_rank() == 0 and args.log_dir is not None and not args.eval:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    dataloader = create_dataset(args)
    diff = diffusion.Diffusion()
    model = diffmae_finetune.DiffMAE(args, diff)
    
    for params in model.parameters():
        print(params)
        break

    if args.finetune:
        checkpoint = torch.load(args.finetune, map_location='cpu')

        print("Load pre-trained checkpoint from: %s" % args.finetune)
        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]
        for k, value in checkpoint_model.copy().items():
            if k[:3] == 'dec':
                del checkpoint_model[k]

        # interpolate position embedding
        interpolate_pos_embed(model, checkpoint_model)

        # load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False)
        # manually initialize fc layer
        trunc_normal_(model.head.weight, std=2e-5)

    if args.cuda:
        args.device = "cuda:{}".format(args.gpu_ids[0])
        if args.multi_gpu:
            model = nn.DataParallel(model, output_device=args.gpu_ids[0], device_ids=args.gpu_ids)
    else:
        args.device = torch.device("cpu")

    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params (M): %.2f' % (n_parameters / 1.e6))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)
    
    param_groups = optim_factory.add_weight_decay(model, args.weight_decay)
    optimizer = torch.optim.Adam(param_groups, lr=args.lr, betas=(0.9, 0.95))
    loss_scaler = NativeScaler()

    criterion = torch.nn.CrossEntropyLoss()

    print("criterion = %s" % str(criterion))

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0

    best_loss = 999999
    for epoch in range(args.start_epoch, args.epochs):

        train_stats = train_one_epoch(
            model, criterion, dataloader['train'],
            optimizer, device, epoch, loss_scaler,
            args.clip_grad,
            log_writer=log_writer,
            args=args
        )

        test_stats = evaluate(dataloader['val'], model, device)
        print(f"Accuracy of the network on the {len(dataloader['val'])} test images: {test_stats['acc1']:.1f}%")
        max_accuracy = max(max_accuracy, test_stats["acc1"])
        print(f'Max accuracy: {max_accuracy:.2f}%')
        
        if test_stats['loss'] < best_loss:
            best_loss = test_stats['loss']
            best_acc1 = test_stats['acc1']
            best_acc5 = test_stats['acc5']
            best_epoch = epoch
            if args.output_dir:
                misc.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch='best')

        if log_writer is not None:
            log_writer.add_scalar('perf/test_acc1', test_stats['acc1'], epoch)
            log_writer.add_scalar('perf/test_acc5', test_stats['acc5'], epoch)
            log_writer.add_scalar('perf/test_loss', test_stats['loss'], epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        **{f'valid_{k}': v for k, v in test_stats.items()},
                        'epoch': epoch,
                        'n_parameters': n_parameters}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.savedir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    args.resume = '{}/checkpoint-best.pth'.format(args.savedir)
    misc.load_model(args=args, model_without_ddp=model, optimizer=optimizer, loss_scaler=loss_scaler)

    with open(os.path.join(args.savedir, "log.txt"), mode="a", encoding="utf-8") as f:
        f.write('Best accuracy: {} at epoch {}'.format(best_acc1, best_epoch) + "\n")

    if args.eval:
        test_stats = evaluate(dataloader['test'], model, device)
        print(f"Accuracy of the network on the {len(dataloader['test'])} test images: {test_stats['acc1']:.1f}%")

        with open(os.path.join(args.savedir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(test_stats) + "\n")
        exit(0)


if __name__ == '__main__':
    args = option.Options().gather_options()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)