# ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware
# Han Cai, Ligeng Zhu, Song Han
# International Conference on Learning Representations (ICLR), 2019.

import os
import sys
_script_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.dirname(_script_dir)
for _p in (_repo_root, _script_dir):
    if _p not in sys.path:
        sys.path.insert(0, _p)


import argparse

from models import ImagenetRunConfig, Cifar10RunConfig, UcmRunConfig, SIRSTRunConfig
from nas_manager import *
from models.super_nets.super_proxyless_SIRST import SuperProxylessNASNets
import numpy as np
from search.models.model_res_Unet import res_UNet
from search.utils.utils import save_path, conv_name_define, save_train_log, decode_gene
from search_train       import search_func, retrain_func, inference


parser = argparse.ArgumentParser()
parser.add_argument('--path',             type=str, default='/media/gfkd/sda/NAS/proxylessnas-master-SIRST-new-final/search/logs')
parser.add_argument('--resume',           action='store_true')
parser.add_argument('--manual_seed',      type=int,  default=0,)
# Retrain-only: Mask R-CNN-style prior on final 1x1 conv bias; see BasicIRSTD_Freq/model/ResUNet/model_ResUNet_init.py:58-62.
# Default off; enable e.g. via proxylessnas/search/retrain_whole_irstd.sh (--final_prior_pi 0.0015).
parser.add_argument(
    '--final_prior_pi',
    type=float,
    default=-1.0,
    help='Retrain only: prior pi for final conv bias (sigmoid prior); outside (0,1) disables (default: off).',
)
parser.add_argument(
    '--retrain_init_ckpt',
    type=str,
    default='',
    help='Retrain only: optional checkpoint path used to initialize network weights before training.',
)

""" run config """
parser.add_argument('--init_lr',          type=float, default=0.01)
parser.add_argument('--lr_schedule_type', type=str,   default='cosine')

'''lr_schedule_param'''
parser.add_argument('--dataset',          type=str,   default='NUAA-SIRST', help='NUAA-SIRST, NUDT-SIRST, IRSTD-SIRST')


parser.add_argument('--train_batch_size', type=int,   default=16)
parser.add_argument('--test_batch_size',  type=int,   default=16)
parser.add_argument('--valid_size',       type=int,   default=1)

parser.add_argument('--opt_type',         type=str,   default='Adagrad', choices=['sgd', 'Adagrad'])
parser.add_argument('--momentum',         type=float, default=0.9)  # opt_param
parser.add_argument('--no_nesterov',      action='store_true')  # opt_param
parser.add_argument('--weight_decay',     type=float, default=4e-5)
parser.add_argument('--label_smoothing',  type=float, default=0.1)
parser.add_argument('--no_decay_keys',    type=str,   default=None, choices=[None, 'bn', 'bn#bias'])

parser.add_argument('--model_init',       type=str,   default='he_fout', choices=['he_fin', 'he_fout'])
parser.add_argument('--init_div_groups',  action='store_true')
parser.add_argument('--validation_frequency', type=int, default=1)
parser.add_argument('--print_frequency',  type=int,   default=10)

parser.add_argument('--n_worker',         type=int,   default=4)
parser.add_argument('--resize_scale',     type=float, default=0.08)
parser.add_argument('--distort_color',    type=str,   default='normal', choices=['normal', 'strong', 'None'])

""" SIRST hyper-parameters """
parser.add_argument('--id_mode',         type=str, default='TXT', help='mode name:  TXT, Ratio')
parser.add_argument('--root',            type=str, default='/media/gfkd/sda/NAS/Nas-SIRST-UNet/dataset')
parser.add_argument('--split_method',    type=str, default='50_50',
                    help='50_50, 10000_100(for NUST-SIRST), 80_20(for IRSTD-SIRST)')
parser.add_argument('--base_size',       type=int, default=256)
parser.add_argument('--crop_size',       type=int, default=256)
parser.add_argument('--suffix',          type=str, default='.png')
parser.add_argument('--eval_batch_size', type=int, default=1)
parser.add_argument('--num_class',       type=int, default=1,  help='model output channel number')

""" net config """
parser.add_argument('--iterations',      type=int, default=5)
parser.add_argument('--conv_type',       type=str, default='res_add')
parser.add_argument('--channel_num',     type=str, default='two', help='one, two, three, four')
parser.add_argument('--in_channel',      type=int, default=3, help='one, two, three, four')


# architecture search config
""" arch search algo and warmup """
parser.add_argument('--arch_algo',       type=str, default='grad', choices=['grad', 'rl'])
# parser.add_argument('--warmup_epochs', type=int, default=300)
# parser.add_argument('--n_epochs',      type=int, default=1500)

""" shared hyper-parameters """
parser.add_argument('--arch_init_type',    type=str,   default='normal', choices=['normal', 'uniform'])
parser.add_argument('--arch_init_ratio',   type=float, default=1e-3)
parser.add_argument('--arch_opt_type',     type=str,   default='adam', choices=['adam'])
parser.add_argument('--arch_lr',           type=float, default=1e-3)
parser.add_argument('--arch_adam_beta1',   type=float, default=0)  # arch_opt_param
parser.add_argument('--arch_adam_beta2',   type=float, default=0.999)  # arch_opt_param
parser.add_argument('--arch_adam_eps',     type=float, default=1e-8)  # arch_opt_param
parser.add_argument('--arch_weight_decay', type=float, default=0)
parser.add_argument('--target_hardware',   type=str,   default='edge-gpu', choices=['mobile', 'cpu', 'gpu', 'flops','params', 'edge-cpu', 'edge-gpu',None])
parser.add_argument('--fast',              type=str,   default='True', help='True, False')

""" Grad hyper-parameters """
parser.add_argument('--grad_update_arch_param_every', type=int, default=5)
parser.add_argument('--grad_update_steps',    type=int,   default=1)
parser.add_argument('--grad_binary_mode',     type=str,   default='full_v2', choices=['full_v2', 'full', 'two', 'darts'])
parser.add_argument('--grad_data_batch',      type=int,   default=None)
parser.add_argument('--grad_reg_loss_type',   type=str,   default='mul#log', choices=['add#linear', 'mul#log', 'add#linear_abs'])  ##original paper is add#linear
parser.add_argument('--grad_reg_loss_lambda', type=float, default=1e-1)  # grad_reg_loss_params
parser.add_argument('--grad_reg_loss_alpha',  type=float, default=0.2)   # grad_reg_loss_params
parser.add_argument('--grad_reg_loss_beta',   type=float, default=0.3)   # grad_reg_loss_params
parser.add_argument('--ref_value',            type=float, default=7 * 1e9)   # grad_reg_loss_params
parser.add_argument('--random_choose',        type=str,   default='False',   help= 'True, False')


""" Retrain parameter """
parser.add_argument('--warmup_epochs',            type=int,   default=0)
parser.add_argument('--n_epochs',                 type=int,   default=0)
parser.add_argument('--retrain_epoch',            type=int,   default=1)
parser.add_argument('--retrain_init_lr',          type=float, default=0.05)
parser.add_argument('--retrain_fixed_lr',         type=float, default=0.01)
parser.add_argument('--retrain_lr_schedule_type', type=str,   default='fixed', help='cosine, fixed')
parser.add_argument('--retrain_valid_size',       type=int,   default=1)
parser.add_argument('--retrain_latency',          type=str,   default='gpu')
parser.add_argument('--gpu',                      type=str,   default='0', help='0,1,2,3')
# parser.add_argument('--retrain_log_path',         type=str,   default='0,1_NUAA-SIRST_Super_all_04_09_2023_23_01_05',
#                                                   help='0,1_NUAA-SIRST_Super_half_01_09_2023_00_09_32,'
#                                                        '0,1_NUAA-SIRST_Super_half_01_09_2023_11_22_49,'
#                                                        '0,1_NUAA-SIRST_Super_29_08_2023_17_23_13,'
#                                                        '0,1_NUAA-SIRST_Super_28_08_2023_22_35_46,'
#                                                        '0,1_NUAA-SIRST_Super_all_04_09_2023_23_01_05，'
#                                                        '')
parser.add_argument('--retrain_log_path',         type=str,   default='0,1_NUAA-SIRST_Super_all_Res_Group_Spa_MBConv_26_10_2023_12_47_26',
                                                  help='0,1_NUAA-SIRST_Super_all_single_17_10_2023_12_44_48,' ### UNet    super all
                                                       '0_NUAA-SIRST_Super_all_single_22_10_2023_12_07_18,'   ### 10-11-34 super all ResNet10
                                                       '0_NUAA-SIRST_Super_all_single_23_10_2023_09_13_49,'   ### 10-11-34 super all ResNet18'
                                                       '0_NUAA-SIRST_Super_all_single_23_10_2023_09_13_49_human_1,'
                                                       '0,1_NUAA-SIRST_Super_all_Res_Group_Spa_MBConv_26_10_2023_18_22_29' ### 20-06-25 super all ResNet18'
                                                                            )

parser.add_argument('--add_decoder',     type=str,   default='False',                  help= 'True, False')
parser.add_argument('--add_encoder0',    type=str,   default='True',                   help= 'True, False')
parser.add_argument('--mode',            type=str,   default='inference',       help= 'search, retrain, all_search_train, inference')  # grad_reg_loss_params
parser.add_argument('--model_name',      type=str,   default='Super_all',              help= 'Super_all,  Super_half_all, Super_half_search, Super_half_retrain')
parser.add_argument('--candidates_type', type=str,   default='single',                 help= 'ResConv,    MBInverted,  MBConvOnly, GroupConv,      SpaConv, '
                                                                                             'DepthSC,    single,      Res_Group_Spa,  Res_Spa_MBConv, whole, whole_all, '
                                                                                             'single_Spa, Res_Group_Spa_MBConv,        Proxyless' )
parser.add_argument('--gene', default='/media/gfkd/sda/NAS/NAS-DNANet-new-share-NEW/Result/search1/0_all_search_1_NUAA-SIRST_DNANet_Meta/phase1_gene.txt',
                    type=str,
                    help='/media/gfkd/sda/NAS/NAS-DNANet-new-new/Result/search1/resnet_18_channel_two_layer_5/phase1_gene_Ushapeder_5_layer_down.txt,'
                         '/media/gfkd/sda/NAS/NAS-DNANet-new-share-NEW/Result/search1/0_all_search_1_NUAA-SIRST_DNANet_22_09_2023_10_11_34/phase1_gene.txt,'
                         '/media/gfkd/sda/NAS/NAS-DNANet-new-share-NEW/Result/search1/0_all_search_1_NUAA-SIRST_DNANet_23_10_2023_20_06_25/phase1_gene.txt,'
                         '/media/gfkd/sda/NAS/NAS-DNANet-new-share-NEW/Result/search1/0_all_search_1_NUAA-SIRST_DNANet_Meta/phase1_gene.txt')

parser.add_argument('--ROC_thr',           type=int, default=10,      help='crop image size')
parser.add_argument('--postprocess',       type=str, default='none',  help='crf, mclc, none')
parser.add_argument('--Inference_resize',  type=str, default='True',  help='True, False')
parser.add_argument('--Inference_repeated',type=int, default=100,     help='100, 200, 400')

if __name__ == '__main__':
    args = parser.parse_args()
    # torch.manual_seed(args.manual_seed)
    # torch.cuda.manual_seed_all(args.manual_seed)
    # np.random.seed(args.manual_seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if args.mode == 'inference':
        # Must match Super_all + iterations=5 skip topology in learned_net/net.config (IRSTD whole_all retrain).
        # The older IRSTD-only phase1_gene has only 4 stages and breaks BFS/load; use the 5-stage gene used for this arch.
        args.gene = '/home/intern/proxylessnas/search1yhy/0_all_search_1_NUAA-SIRST_DNANet_23_10_2023_20_06_25_new/phase1_gene.txt'

    gene_path     = args.gene.split('phase1_gene')[0]+'train_parameters.txt'
    args.backbone, args.channel_num, args.iterations, args.conv_type, args.add_decoder, args.add_encoder0, args.more_down =  decode_gene(gene_path)

    if   args.mode == 'search':
        search_log = search_func(args)

    elif args.mode == 'retrain':
        search_log = args.path + '/'+args.retrain_log_path
        inference_path = retrain_func(args, search_log)
        search_log = inference_path.split('_Retrain')[0]
        inference(args, search_log, inference_path)

    elif args.mode == 'all_search_train':
        if args.model_name   == 'Super_half_all' :
            args.model_name  =  'Super_half_search'
            search_log       =   search_func(args)
            args.model_name  =  'Super_half_retrain'
            retrain_func(args, search_log)

        elif args.model_name == 'Super_all' :
            search_log       =   search_func(args)
            inference_path   =   retrain_func(args, search_log)
            search_log       =   inference_path.split('_Retrain')[0]
            inference(args, search_log, inference_path)

    elif args.mode == 'inference':
        # args.gene = '/media/gfkd/sda/NAS/NAS-DNANet-new-share-NEW/Result/search1/0_all_search_1_NUAA-SIRST_DNANet_23_10_2023_20_06_25/phase1_gene.txt'

        # args.dataset = 'NUAA-SIRST-Old' ### 如果是old的话，归一化用的就是老的均值和标准差
        ### Res_3*3: None Resize   75.25%  95.43%  26.16(-6)
        ### Res_3*3:      Resize   75.62%  96.19%  9.911(-6)
        # inference_path = '/media/gfkd/sda/NAS/proxylessnas-master-SIRST-new-final/search/logs/0_NUAA-SIRST_Super_all_3x3_ResConv_12_11_2023_19_57_11_Retrain'


        ### FLOPs: None Resize   75.05%  95.43%  11.12(-6)
        # inference_path = '/media/gfkd/sda/NAS/proxylessnas-master-SIRST-new-final/search/logs/0,1_NUAA-SIRST_Super_all_Res_Group_Spa_MBConv_01_11_2023_11_34_58_Retrain_2'

        ### CPU(NUAA): None Resize   75.85%  96.95%  36.86(-6)
        # inference_path = '/media/gfkd/sda/NAS/proxylessnas-master-SIRST-new-final/search/logs/0,1_NUAA-SIRST_Super_all_Res_Group_Spa_MBConv_26_10_2023_12_47_26_Retrain'
        ### CPU(IRSTD):
        # args.dataset = 'IRSTD-SIRST' ### 如果是old的话，归一化用的就是老的均值和标准差
        # inference_path = '/media/gfkd/sda/NAS/proxylessnas-master-SIRST-new-final/search/logs/0,1_NUAA-SIRST_Super_all_Res_Group_Spa_MBConv_26_10_2023_12_47_26_Retrain_15'

        ### GPU:   None Resize   74.34%  96.57%  39.71(-6)
        # inference_path = '/media/gfkd/sda/NAS/proxylessnas-master-SIRST-new-final/search/logs/0,1_NUAA-SIRST_Super_all_Res_Group_Spa_MBConv_27_10_2023_10_47_00_Retrain_4'
        ### GPU(IRSTD):
        # args.dataset = 'IRSTD-SIRST'
        # inference_path = '/media/gfkd/sda/NAS/proxylessnas-master-SIRST-new-final/search/logs/0,1_NUAA-SIRST_Super_all_Res_Group_Spa_MBConv_27_10_2023_10_47_00_Retrain_7'

        ### E-CPU: None Resize   73.49%  95.81%  39.28(-6)
        ### E-CPU:      Resize   73.46%  96.19%  19.53(-6)
        # inference_path = '/media/gfkd/sda/NAS/proxylessnas-master-SIRST-new-final/search/logs/0,1_NUAA-SIRST_Super_all_Res_Group_Spa_MBConv_16_11_2023_00_20_25_Retrain'

        ### E-GPU: None Resize   76.20%  95.81%  21.67(-6)
        ### E-GPU:      Resize   74.34%  96.57%  39.71(-6)
        # inference_path = '/media/gfkd/sda/NAS/proxylessnas-master-SIRST-new-final/search/logs/0,1_NUAA-SIRST_Super_all_Res_Group_Spa_MBConv_13_11_2023_00_37_52_Retrain'

        ### Proxyless:
        # inference_path = '/media/gfkd/sda/NAS/proxylessnas-master-SIRST-new-final/search/logs/0,1_NUAA-SIRST_Super_all_MBInverted_24_10_2023_04_53_10_Retrain'

        ### 6,7_IRSTD-SIRST Super_all whole_all irstd1k Retrain_8
        args.dataset = 'IRSTD-SIRST'
        args.root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'datasetyhy')
        args.split_method = '80_20'
        inference_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'logs',
            '6,7_IRSTD-SIRST_Super_all_whole_all_31_03_2026_20_54_05_irstd1k_Retrain_8',
        )

        search_log     = inference_path.split('_Retrain')[0]
        inference(args, search_log, inference_path)
