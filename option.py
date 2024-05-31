"""
This code is based from
https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
"""

import os
import json
import random
import datetime
import argparse
import numpy as np

import torch


class Options():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
        parser.add_argument('--epochs', default=100, type=int)
        parser.add_argument('--save_freq', default=10, type=int)

        # Model parameters
        parser.add_argument('--model', default='diffmae', type=str, metavar='MODEL',
                            help='Name of model to train')
        parser.add_argument('--depth', default=8, type=int)
        parser.add_argument('--num_heads', default=8, type=int)
        parser.add_argument('--img_size', default=224, type=int,
                            help='input image size')
        parser.add_argument('--mask_ratio', default=0.75, type=float,
                            help='Masking ratio (percentage of removed patches).')
        parser.add_argument('--norm_pix_loss', action='store_true',
                            help='Use (per-patch) normalized points as targets for computing loss')
        parser.set_defaults(norm_pix_loss=False)

        # Optimizer parameters
        parser.add_argument('--weight_decay', type=float, default=0.05,
                            help='weight decay (default: 0.05)')
        parser.add_argument('--layer_decay', type=float, default=0.75,
                        help='layer-wise lr decay from ELECTRA/BEiT')
        parser.add_argument('--lr', type=float, default=None, metavar='LR',
                            help='learning rate (absolute lr)')
        parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                            help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
        parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                            help='lower lr bound for cyclic schedulers that hit 0')
        parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                            help='epochs to warmup LR')
        parser.add_argument('--clip_grad', type=float, default=0.,
                            help='clip grad')

        # Dataset parameters
        parser.add_argument('--patch_size', default=8, type=int)
        parser.add_argument('--data_path', default='../../data/', type=str,
                            help='dataset path')
        parser.add_argument('--dataset', type=str, required=True,
                            help='name of dataset')
        parser.add_argument('--output_dir', default='./output_dir',
                            help='path where to save, empty for no saving')
        parser.add_argument('--log_dir', default='./output_dir',
                            help='path where to tensorboard log')

        parser.add_argument('--manual_seed', default=0, type=int)
        parser.add_argument('--finetune', default='',
                            help='model path for finetuning')
        parser.add_argument('--resume', default='',
                            help='resume from checkpoint')
        parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')

        parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                            help='start epoch')
        parser.add_argument('--num_workers', default=0, type=int)
        parser.add_argument('--pin_mem', action='store_true',
                            help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
        parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
        parser.set_defaults(pin_mem=True)

        parser.add_argument('--n_channels', type=int, default=3,
                            help='number of features, default=3')
        parser.add_argument('--n_classes', type=int, default=10,
                            help='number of classes, default=10')
        
        # Training parameters
        parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
        parser.add_argument('--mode', type=str, choices=['pretrain', 'finetune'])
        parser.add_argument('--emb_dim', type=int, default=1024,
                            help='feature dimension for embedding')
        parser.add_argument('--dec_emb_dim', type=int, default=512,
                            help='feature dimension for decoder embedding')

        parser.add_argument('--cuda', action='store_true', help='enables cuda')
        parser.add_argument('--multi_gpu', action='store_true', help='enables multi gpu')
        parser.add_argument('--gpu_ids', type=int, nargs='+', default=[],
            help='gpu ids: e.g. 0  0,1,2, 0,2. use [] for CPU')

        parser.add_argument('--dist_on_itp', action='store_true')

        parser.add_argument('--sampling', action='store_true')
        self.initialized = True
        return parser

    def save_options(self, args):
        note = input("Anything to note: ")

        os.makedirs(args.savedir, exist_ok=True)
        os.makedirs('{}/sample'.format(args.savedir, exist_ok=True))
        config_file = args.savedir + "/config.txt"
        with open(config_file, 'w') as f:
            json.dump(args.__dict__, f, indent=2)
            f.write("\nnote: {}\n".format(
                note
            ))

    def setup(self, args):
        try:
            os.makedirs(args.output_dir)
        except OSError:
            pass

        if args.manual_seed is None:
            args.manual_seed = random.randint(1, 10000)

        print("Random Seed: ", args.manual_seed)
        random.seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)

        torch.backends.cudnn.benchmark = True

        if args.cuda:
            torch.cuda.manual_seed_all(args.manual_seed)

        dt = datetime.datetime.now()
        date = dt.strftime("%Y%m%d-%H%M")

        model_opt = args.dataset  + "-" + date + "-" + args.model
        
        args.savedir = os.path.join(args.output_dir, model_opt)
        os.makedirs(args.savedir, exist_ok=True)

        args.log_file = os.path.join(args.savedir, 'log.csv')

    def set_device(self, args):
        n_gpu = torch.cuda.device_count()
        if args.multi_gpu and len(args.gpu_ids) == 0 and torch.cuda.is_available():
            args.gpu_ids = list(range(n_gpu))
        elif args.gpu_ids and torch.cuda.is_available():
            gpu_ids = args.gpu_ids
            args.gpu_ids = []
            for id in gpu_ids:
                if id >= 0 and id < n_gpu:
                    args.gpu_ids.append(id)
            args.gpu_ids = sorted(args.gpu_ids)
            if len(args.gpu_ids) > 1:
                args.multi_gpu = True
            else:
                args.multi_gpu = False        
        else:
            args.gpu_ids = []

        if args.cuda:
            args.device = "cuda:{}".format(args.gpu_ids[0])
        else:
            args.device = "cpu"

    def gather_options(self):
        if not self.initialized:
            parser = argparse.ArgumentParser()
            parser = self.initialize(parser)

        self.parser = parser
        args = parser.parse_args()
        self.setup(args)
        self.set_device(args)
        print(args)
        self.save_options(args)

        return args
