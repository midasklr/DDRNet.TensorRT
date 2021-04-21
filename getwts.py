# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

import argparse
import os
import pprint
import shutil
import sys

import logging
import time
import timeit
from pathlib import Path
import time
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import struct
import _init_paths
import models
import cv2
import torch.nn.functional as F
import datasets
from config import config
from config import update_config
from core.function import testval, test
from utils.modelsummary import get_model_summary
from utils.utils import create_logger, FullModel, speed_test

def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')
    
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default="experiments/cityscapes/ddrnet23_slim.yaml",
                        type=str)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    update_config(config, args)

    return args

def main():
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
    args = parse_args()

    logger, final_output_dir, _ = create_logger(
        config, args.cfg, 'test')

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    # build model
    if torch.__version__.startswith('1'):
        module = eval('models.'+config.MODEL.NAME)
        module.BatchNorm2d_class = module.BatchNorm2d = torch.nn.BatchNorm2d
    model = eval('models.'+config.MODEL.NAME +
                 '.get_seg_model')(config)

    dump_input = torch.rand(
        (1, 3, config.TRAIN.IMAGE_SIZE[1], config.TRAIN.IMAGE_SIZE[0])
    )
    logger.info(get_model_summary(model.cuda(), dump_input.cuda()))

    if config.TEST.MODEL_FILE:
        model_state_file = config.TEST.MODEL_FILE
    else:
        # model_state_file = os.path.join(final_output_dir, 'best_0.7589.pth')
        model_state_file = os.path.join(final_output_dir, 'best.pth')    
    logger.info('=> loading model from {}'.format(model_state_file))
        
    pretrained_dict = torch.load('model_best_bacc.pth.tar')
    if 'state_dict' in pretrained_dict:
        pretrained_dict = pretrained_dict['state_dict']
    newstate_dict = {k:v for k,v in pretrained_dict.items() if k in model.state_dict()}
    #  print(pretrained_dict.keys())

    model.load_state_dict(newstate_dict)
    model = model.cuda()

    if True:
        save_wts = True
        print(model)
        if save_wts:
            f = open('DDRNet_CS.wts', 'w')
            f.write('{}\n'.format(len(model.state_dict().keys())))
            for k, v in model.state_dict().items():
                print("Layer {} ; Size {}".format(k,v.cpu().numpy().shape))
                vr = v.reshape(-1).cpu().numpy()
                f.write('{} {} '.format(k, len(vr)))
                for vv in vr:
                    f.write(' ')
                    f.write(struct.pack('>f', float(vv)).hex())
                f.write('\n')



if __name__ == '__main__':
    main()
