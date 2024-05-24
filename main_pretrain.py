# main implementation code for pre-training

import os
import re
import glob
import time
import json
import datetime
import torch
import torch.nn as nn

import timm

assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler

import option
from dataset import create_dataset
from model import diffmae, diffusion
from engine_pretrain import train_one_epoch, evaluate

def main(args):
    
    if args.mode != 'pretrain':
        print('Pre-training phase: args.mode has to be "pretrain"')
        exit(0)
    
    dataloader = create_dataset(args)

    diff = diffusion.Diffusion(schedule='cosine')
    model = diffmae.DiffMAE(args, diff)

    if args.cuda:
        args.device = "cuda:{}".format(args.gpu_ids[0])
        if args.multi_gpu:
            model = nn.DataParallel(model, output_device=args.gpu_ids[0], device_ids=args.gpu_ids)
    else:
        args.device = torch.device("cpu")

    model = model.to(args.device)  
    eff_batch_size = args.batch_size * args.accum_iter
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)
    print("effective batch size: %d" % eff_batch_size)
    
    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.add_weight_decay(model, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))

    loss_scaler = NativeScaler()

    if args.resume: # 전에 저장된 checkpoint부터 이어서 학습하고싶은 경우 실행
        misc.load_model(args=args, model_without_ddp=model, optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    log_writer = None
    # for epoch in range(args.start_epoch, args.epochs):
    #     train_stats = train_one_epoch(
    #         model, dataloader['train'],
    #         optimizer, epoch, loss_scaler,
    #         log_writer=log_writer,
    #         args=args, iter=epoch
    #     )
    #     if args.output_dir and (epoch % args.save_freq == 0 or epoch + 1 == args.epochs):
    #         misc.save_model(
    #             args=args, model=model, model_without_ddp=model, optimizer=optimizer,
    #             loss_scaler=loss_scaler, epoch=epoch)

    #     log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
    #                     'epoch': epoch,}

    #     if args.output_dir and misc.is_main_process():
    #         if log_writer is not None:
    #             log_writer.flush()
    #         with open(os.path.join(args.savedir, "log.txt"), mode="a", encoding="utf-8") as f:
    #             f.write(json.dumps(log_stats) + "\n")

    # total_time = time.time() - start_time
    # total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    # print('Training time {}'.format(total_time_str))

    # pattern = re.compile(r'\d+')
    # file_list = sorted(glob.glob('{}/*.pth'.format(args.savedir)), key=lambda x:int(pattern.findall(x)[-1]))
    # args.resume = file_list[-1]
    args.resume = 'F:\kny\workspace\dts\diffmae-pytorch\output_dir\cifar10-20240523-2308-diffmae\checkpoint-99.pth'
    misc.load_model(args=args, model_without_ddp=model, optimizer=optimizer, loss_scaler=loss_scaler)

    if args.eval:
        test_stats = evaluate(model, dataloader['test'], epoch='', args=args)

        log_stats = {**{f'test_{k}': v for k, v in test_stats.items()}}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.savedir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

if __name__ == '__main__':
    args = option.Options().gather_options()
    main(args)