# ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware
# Han Cai, Ligeng Zhu, Song Han
# International Conference on Learning Representations (ICLR), 2019.

import argparse

from   models      import ImagenetRunConfig, Cifar10RunConfig, UcmRunConfig, SIRSTRunConfig
from   nas_manager import *
from   models.super_nets.super_proxyless_SIRST import SuperProxylessNASNets
import numpy as np
import os
from search.models.model_res_Unet import res_UNet
from search.utils.utils import save_path,conv_name_define,save_train_log
from search_train       import search_func

# ref values
ref_values = {
    'flops': {
        '16.00': 59 * 1e6,
        '32.00': 97 * 1e6,
        '48.00': 209 * 1e6,
    },
    # ms
    'mobile': {
        '16.00': 80,
    },
    'cpu': {'16.00': 80},
    'gpu': {'16.00': 40,
            '32.00': 80},
}

parser = argparse.ArgumentParser()
parser.add_argument('--path',                type    = str, default='/media/gfkd/sda/NAS/proxylessnas-master-SIRST-new/search/logs')
parser.add_argument('--resume',              action  = 'store_true')
parser.add_argument('--manual_seed',         default = 0, type=int)

""" run config """
parser.add_argument('--init_lr',           type=float, default=0.01)
parser.add_argument('--lr_schedule_type',  type=str,   default='cosine')


'''lr_schedule_param'''
parser.add_argument('--dataset',           type=str, default='NUAA-SIRST', choices=['imagenet','cifar10','ucm', 'NUAA-SIRST'])


parser.add_argument('--train_batch_size',  type=int, default=16)
parser.add_argument('--test_batch_size',   type=int, default=16)
parser.add_argument('--valid_size',        type=int, default=1)

parser.add_argument('--opt_type',             type=str,   default='Adagrad', choices=['sgd', 'Adagrad'])
parser.add_argument('--momentum',             type=float, default=0.9)  # opt_param
parser.add_argument('--no_nesterov',          action='store_true')  # opt_param
parser.add_argument('--weight_decay',         type=float, default=4e-5)
parser.add_argument('--label_smoothing',      type=float, default=0.1)
parser.add_argument('--no_decay_keys',        type=str,   default=None, choices=[None, 'bn', 'bn#bias'])

parser.add_argument('--model_init',           type=str,   default='he_fout', choices=['he_fin', 'he_fout'])
parser.add_argument('--init_div_groups',      action='store_true')
parser.add_argument('--validation_frequency', type=int,   default=1)
parser.add_argument('--print_frequency',      type=int,   default=10)

parser.add_argument('--n_worker',      type=int,   default=4)
parser.add_argument('--resize_scale',  type=float, default=0.08)
parser.add_argument('--distort_color', type=str,   default='normal', choices=['normal', 'strong', 'None'])

""" SIRST hyper-parameters """
parser.add_argument('--mode',            type=str, default='TXT',   help='mode name:  TXT, Ratio')
parser.add_argument('--root',            type=str, default='/media/gfkd/sda/NAS/Nas-SIRST-UNet/dataset')
parser.add_argument('--split_method',    type=str, default='50_50', help='50_50, 10000_100(for NUST-SIRST), 80_20(for IRSTD-SIRST)')
parser.add_argument('--base_size',       type=int, default=256)
parser.add_argument('--crop_size',       type=int, default=256)
parser.add_argument('--suffix',          type=str, default='.png')
parser.add_argument('--eval_batch_size', type=int, default=1)
parser.add_argument('--num_class',       type=int, default=1, help='model output channel number')

""" net config """
# parser.add_argument('--width_stages',  type=str,   default='24,40,80,96,192,320')
# parser.add_argument('--n_cell_stages', type=str,   default='4,4,4,4,4,1')
# parser.add_argument('--stride_stages', type=str,   default='2,2,2,1,2,1')

# architecture search config
""" arch search algo and warmup """
parser.add_argument('--arch_algo',         type=str, default='grad', choices=['grad', 'rl'])
parser.add_argument('--warmup_epochs',     type=int, default=1500)
parser.add_argument('--n_epochs',          type=int, default=0)

""" shared hyper-parameters """
parser.add_argument('--arch_init_type',    type=str,   default='normal', choices=['normal', 'uniform'])
parser.add_argument('--arch_init_ratio',   type=float, default=1e-3)
parser.add_argument('--arch_opt_type',     type=str,   default='adam', choices=['adam'])
parser.add_argument('--arch_lr',           type=float, default=1e-3)
parser.add_argument('--arch_adam_beta1',   type=float, default=0)      # arch_opt_param
parser.add_argument('--arch_adam_beta2',   type=float, default=0.999)  # arch_opt_param
parser.add_argument('--arch_adam_eps',     type=float, default=1e-8)   # arch_opt_param
parser.add_argument('--arch_weight_decay', type=float, default=0)


""" Grad hyper-parameters """
parser.add_argument('--grad_update_arch_param_every', type=int,    default=5)
parser.add_argument('--grad_update_steps',            type=int,    default=1)
parser.add_argument('--grad_binary_mode',             type=str,    default='full_v2', choices=['full_v2', 'full', 'two', 'darts'])
parser.add_argument('--grad_data_batch',              type=int,    default=None)
parser.add_argument('--grad_reg_loss_type',           type=str,    default='mul#log', choices=['add#linear', 'mul#log'])  ##original paper is add#linear
parser.add_argument('--grad_reg_loss_lambda',         type=float,  default=1e-1)  # grad_reg_loss_params
parser.add_argument('--grad_reg_loss_alpha',          type=float,  default=0.2)   # grad_reg_loss_params
parser.add_argument('--grad_reg_loss_beta',           type=float,  default=0.3)   # grad_reg_loss_params

parser.add_argument('--gene', default='/media/gfkd/sda/NAS/NAS-DNANet-new/Phase1_gene_save/phase1_gene_Ushapeder.txt',
                              type=str,
                              help='/media/gfkd/sda/NAS/NAS-DNANet-new/Phase1_gene_save/phase1_gene_Ushapeder.txt,')
parser.add_argument('--gpu',             type=str, default='0', help='0,1,2,3,'
                                                                       '0', )

parser.add_argument('--model_name',      type=str,   default='Super_half_search', help='Super_all, Super_half_search, Super_half_retrain')
parser.add_argument('--candidates_type', type=str,   default='single',     help='ResConv, MBInverted, GroupConv,     SpaConv, '
                                                                                 'DepthSC, single,     Res_Group_Spa, Res_Spa_MBConv, whole, whole_all, ' )
parser.add_argument('--repeat_num',        type=int,   default=1)
parser.add_argument('--iterations',        type=int,   default=5)
parser.add_argument('--conv_type',         type=str,   default='res_add')
parser.add_argument('--channel_num',       type=int,   default=32)
parser.add_argument('--target_hardware',   type=str,   default='flops', choices=['mobile', 'cpu', 'gpu', 'flops', None])
parser.add_argument('--fast',              type=str,   default='True',    help='True, False')

if __name__ == '__main__':
    args = parser.parse_args()
    torch.manual_seed(args.manual_seed)
    torch.cuda.manual_seed_all(args.manual_seed)
    np.random.seed(args.manual_seed)


    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    search_func(args)
