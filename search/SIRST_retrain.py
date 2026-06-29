# ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware
# Han Cai, Ligeng Zhu, Song Han
# International Conference on Learning Representations (ICLR), 2019.

import argparse
import numpy as np
import os
import json

import torch

from models import *
from run_manager import RunManager
from search.utils.utils import save_path
from search_train import retrain_func

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path',             type=str,   default='/media/gfkd/sda/NAS/proxylessnas-master-SIRST-new/search/logs')
    parser.add_argument('--train',            type=bool,  default=True)

    parser.add_argument('--manual_seed',      type=int,   default=0)
    parser.add_argument('--resume',           action='store_true')



    """ lr_schedule_param """
    parser.add_argument('--dataset',           type=str, default='NUAA-SIRST', choices=['imagenet','cifar10','ucm','NUAA-SIRST'])
    parser.add_argument('--train_batch_size',  type=int, default=16)
    parser.add_argument('--test_batch_size',   type=int, default=16)

    parser.add_argument('--opt_type',         type=str,   default='sgd', choices=['sgd'])
    parser.add_argument('--momentum',         type=float, default=0.9)  # opt_param
    parser.add_argument('--no_nesterov',      action='store_true')  # opt_param
    parser.add_argument('--weight_decay',     type=float, default=4e-5)
    parser.add_argument('--label_smoothing',  type=float, default=0.1)
    parser.add_argument('--no_decay_keys',    type=str,   default='bn', choices=['None', 'bn', 'bn#bias'])

    parser.add_argument('--model_init',       type=str,   default='he_fout', choices=['he_fin', 'he_fout'])
    parser.add_argument('--init_div_groups',  action='store_true')
    parser.add_argument('--validation_frequency', type=int, default=1)
    parser.add_argument('--print_frequency',  type=int,   default=10)

    parser.add_argument('--n_worker',         type=int,   default=4)
    parser.add_argument('--resize_scale',     type=float, default=0.08)
    parser.add_argument('--distort_color',    type=str,   default='None', choices=['normal', 'strong', 'None'])

    """ SIRST hyper-parameters """
    parser.add_argument('--mode',             type=str,   default='TXT',   help='mode name:  TXT, Ratio')


    """ net config """
    parser.add_argument('--bn_momentum',      type=float, default=0.1)
    parser.add_argument('--bn_eps',           type=float, default=1e-3)
    parser.add_argument('--net',              type=str,   default='proxyless_gpu',
                        choices=['proxyless_gpu', 'proxyless_cpu', 'proxyless_mobile', 'proxyless_mobile_14'])
    parser.add_argument('--dropout',          type=float, default=0)

    # parser.add_argument('--model_name',               type=str,   default='Super_all', help='Super_all, Super_half')
    parser.add_argument('--retrain_epoch',            type=int,   default=1500)
    parser.add_argument('--retrain_valid_size',       type=int,   default=1)
    parser.add_argument('--retrain_lr_schedule_type', type=str,   default='fixed', help='cosine, fixed')
    parser.add_argument('--retrain_latency',          type=str,   default='gpu')
    parser.add_argument('--retrain',                  type=bool,  default=True)
    parser.add_argument('--retrain_fixed_lr',         type=float, default=0.01)
    parser.add_argument('--gpu',                      type=str,   default='0', help='0,1,2,3,'
                                                                                    '0',)

    parser.add_argument('--retrain_log_path', type=str,  default='0_NUAA-SIRST_Super_half_search_single_20_09_2023_00_02_29',
                        help='0,1_NUAA-SIRST_Super_half_search_Res_Group_Spa_19_09_2023_16_02_12,'
                             '0_NUAA-SIRST_Super_half_search_single_20_09_2023_00_02_29')


    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args    = parse_args()

    torch.manual_seed(args.manual_seed)
    torch.cuda.manual_seed_all(args.manual_seed)
    np.random.seed(args.manual_seed)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    search_log = '/media/gfkd/sda/NAS/proxylessnas-master-SIRST-new/search/logs/'+args.retrain_log_path
    retrain_func(args,search_log)

