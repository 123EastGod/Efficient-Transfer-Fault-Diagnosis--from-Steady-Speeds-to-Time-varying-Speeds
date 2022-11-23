#!/usr/bin/python
# -*- coding:utf-8 -*-
import argparse
import os
from datetime import datetime
from utils.logger import setlogger
import logging
from utils.train_utils_combines import train_utils
import torch
import warnings
print(torch.__version__)
warnings.filterwarnings('ignore')
args = None

def parse_args():
    parser = argparse.ArgumentParser(description='Train')

    parser.add_argument('--model_name', type=str, default='cnn_features_1d', help='the name of the model')
    parser.add_argument('--data_name', type=str, default='CWRU', help='the name of the data')
    parser.add_argument('--data_dir', type=str, default='D:\CWRU', help='the directory of the data')
    parser.add_argument('--transfer_task', type=list, default=[[1], [2]], help='transfer learning tasks')#TODO TASKs
    parser.add_argument('--normlizetype', type=str, default='0-1', help='nomalization type')

    parser.add_argument('--cuda_device', type=str, default='0', help='assign device')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint', help='the directory to save the model')
    parser.add_argument("--pretrained", type=bool, default=False, help='whether to load the pretrained model')
    parser.add_argument('--batch_size', type=int, default=64, help='batchsize of the training process')#TODO batch
    parser.add_argument('--num_workers', type=int, default=0, help='the number of training process')

    parser.add_argument('--bottleneck', type=bool, default=True, help='whether using the bottleneck layer')
    parser.add_argument('--bottleneck_num', type=int, default=256, help='whether using the bottleneck layer')
    parser.add_argument('--last_batch', type=bool, default=False, help='whether using the last batch')
    parser.add_argument('--hidden_size', type=int, default=1024, help='whether using the last batch')

    parser.add_argument('--distance_metric', type=bool, default=True, help='whether use distance metric')
    parser.add_argument('--distance_loss', type=str, choices=['MK-MMD', 'Cauthy'], default='Cauthy')
    parser.add_argument('--trade_off_distance', type=str, default='Step', help='')
    parser.add_argument('--lam_distance', type=float, default=1, help='this is used for Cons')

    parser.add_argument('--opt', type=str, choices=['sgd', 'adam'], default='adam', help='the optimizer')
    parser.add_argument('--lr', type=float, default=1e-3, help='the initial learning rate')#TODO lr
    parser.add_argument('--momentum', type=float, default=0.9, help='the momentum for sgd')
    parser.add_argument('--weight-decay', type=float, default=1e-5, help='the weight decay')
    parser.add_argument('--lr_scheduler', type=str, choices=['step', 'exp', 'stepLR'], default='step', help='the learning rate schedule')
    parser.add_argument('--gamma', type=float, default=0.1, help='learning rate scheduler parameter for step and exp')
    parser.add_argument('--steps', type=str, default='150, 250', help='the learning rate decay for step and stepLR')

    parser.add_argument('--middle_epoch', type=int, default=10, help='max number of epoch')
    parser.add_argument('--max_epoch', type=int, default=200, help='max number of epoch')#TODO max_epoch
    parser.add_argument('--print_step', type=int, default=50, help='the interval of log training information')

    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device.strip()
    sub_dir = args.model_name + '_' + datetime.strftime(datetime.now(), '%m%d-%H%M%S')
    save_dir = os.path.join(args.checkpoint_dir, sub_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    setlogger(os.path.join(save_dir, 'train.log'))

    for k, v in args.__dict__.items():
        logging.info("{}: {}".format(k, v))

    trainer = train_utils(args, save_dir)
    trainer.setup()
    trainer.train()





