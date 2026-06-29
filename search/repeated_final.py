import os

# #### 01-05-54 70.2%   #### 01-42-23 70.3%
# os.system('python  SIRST_main_all.py    --target_hardware flops   --gpu 0,1 '
#           '--warmup_epochs 1            --n_epochs 2              --retrain_epoch 1500 '
#           '--model_name Super_half_all  --mode all_search_train   --candidates_type single  --fast False '
#           '--gene /media/gfkd/sda/NAS/NAS-DNANet-new-new/Result/search1/0_all_search_1_NUAA-SIRST_DNANet_24_09_2023_20_29_53/phase1_gene.txt ')


##########################所有的candidate貌似训练不动，首先需要调整一下学习率试下，然后再用小的candidate试下。
# #### 23-58-25 70.5%
# os.system('python  SIRST_main_all.py    --target_hardware flops   --gpu 0,1 '
#           '--warmup_epochs 1            --n_epochs 2              --retrain_epoch 1500 '
#           '--model_name Super_all       --mode all_search_train   --candidates_type single  --fast False '
#           '--gene /media/gfkd/sda/NAS/NAS-DNANet-new-new/Result/search1/0_all_search_1_NUAA-SIRST_DNANet_24_09_2023_20_29_53/phase1_gene.txt ')

# #### 00-45-40 70.3%
# os.system('python  SIRST_main_all.py    --target_hardware flops   --gpu 0,1 '
#           '--warmup_epochs 1            --n_epochs 2              --retrain_epoch 1500 '
#           '--model_name Super_all       --mode all_search_train   --candidates_type single  --fast False '
#           '--gene /media/gfkd/sda/NAS/NAS-DNANet-new-new/Result/search1/0_all_search_1_NUAA-SIRST_DNANet_24_09_2023_20_29_53/phase1_gene.txt ')

# #### 01-34-37 cpu:5.31-5.25ms  71.9%
# os.system('python  SIRST_main_all.py    --target_hardware cpu     --gpu 0,1 '
#           '--warmup_epochs 300          --n_epochs 1500            --retrain_epoch 1500 '
#           '--model_name Super_all       --mode all_search_train   --candidates_type whole  --fast False '
#           '--gene /media/gfkd/sda/NAS/NAS-DNANet-new-new/Result/search1/0_all_search_1_NUAA-SIRST_DNANet_22_09_2023_18_24_56/phase1_gene.txt ')

# #### 08-04-10 cpu:4.32-4.27ms  70.8%
# os.system('python  SIRST_main_all.py    --target_hardware cpu     --gpu 0,1 '
#           '--warmup_epochs 300          --n_epochs 1500            --retrain_epoch 1500 '
#           '--model_name Super_all       --mode all_search_train   --candidates_type whole  --fast False '
#           '--gene /media/gfkd/sda/NAS/NAS-DNANet-new-new/Result/search1/0_all_search_1_NUAA-SIRST_DNANet_22_09_2023_18_24_56/phase1_gene.txt ')


#### 14-21-37 gpu:1.01-0.99ms  Retrain error
# os.system('python  SIRST_main_all.py    --target_hardware gpu   --gpu 0,1 '
#           '--warmup_epochs 300          --n_epochs 1500            --retrain_epoch 1500 '
#           '--model_name Super_all       --mode all_search_train   --candidates_type whole  --fast False '
#           '--gene /media/gfkd/sda/NAS/NAS-DNANet-new-new/Result/search1/0_all_search_1_NUAA-SIRST_DNANet_22_09_2023_18_24_56/phase1_gene.txt ')


# #### 15-34-53 error
# os.system('python  SIRST_main_all.py    --target_hardware gpu   --gpu 0,1 '
#           '--warmup_epochs 300          --n_epochs 1500            --retrain_epoch 1500 '
#           '--model_name Super_all       --mode all_search_train   --candidates_type whole  --fast False '
#           '--gene /media/gfkd/sda/NAS/NAS-DNANet-new-new/Result/search1/0_all_search_1_NUAA-SIRST_DNANet_22_09_2023_18_24_56/phase1_gene.txt ')



# ### 试一下减少candiate item   还是没有实现target aware
# #### 22-39-48  cpu:21.78->22.18   Search:68.1%   Retrain: 72.6%
# os.system('python  SIRST_main_all.py    --target_hardware cpu     --gpu 0,1 '
#           '--warmup_epochs 300          --n_epochs 1500            --retrain_epoch 1500 '
#           '--model_name Super_all       --mode all_search_train   --candidates_type ResConv  --fast False --arch_lr 0.01 '
#           '--gene /media/gfkd/sda/NAS/NAS-DNANet-new-new/Result/search1/0_all_search_1_NUAA-SIRST_DNANet_22_09_2023_18_24_56/phase1_gene.txt ')
#
#
# ### 试一下提高LR              还是没有实现target aware
# #### 05-43-50  cpu:21.3->20.85   Search:67.4%   Retrain: 68.3%
# os.system('python  SIRST_main_all.py    --target_hardware cpu     --gpu 0,1 '
#           '--warmup_epochs 300          --n_epochs 1500            --retrain_epoch 1500 '
#           '--model_name Super_all       --mode all_search_train   --candidates_type ResConv  --fast False --arch_lr 0.05 '
#           '--gene /media/gfkd/sda/NAS/NAS-DNANet-new-new/Result/search1/0_all_search_1_NUAA-SIRST_DNANet_22_09_2023_18_24_56/phase1_gene.txt ')


#### Res18 channel two, 自适应channel  无 short cut  12-44-48 R1: 73.2%   R2: 72.7%  R3: 72.0%  R4: 73.4%  R5: 74.4%   AVG: 73.14%
# os.system('python  SIRST_main_all.py    --target_hardware flops   --gpu 0 '
#           '--warmup_epochs 0            --n_epochs 0              --retrain_epoch 1500 '
#           '--model_name Super_all       --mode retrain            --candidates_type single  --fast False --arch_lr 0.005 '
#           '--retrain_log_path   0,1_NUAA-SIRST_Super_all_single_17_10_2023_12_44_48')


# #### Res18 channel two, 自适应channel 有shortcut 12-44-48   R6: 72.9%   R7: 74.1%  R8: 73.7%  R9: 74.0%    AVG: 73.67%
# os.system('python  SIRST_main_all.py    --target_hardware flops   --gpu 0 '
#           '--warmup_epochs 0            --n_epochs 0              --retrain_epoch 1500 '
#           '--model_name Super_all       --mode retrain            --candidates_type single  --fast False --arch_lr 0.005 '
#           '--retrain_log_path   0,1_NUAA-SIRST_Super_all_single_17_10_2023_12_44_48')


# #### Res18 channel two, 自适应channel 有shortcut Xavier初始化  R10：75.2%  R11：72.2%  R12：73.3%  R13：73.1%  AVG:73.45%
# os.system('python  SIRST_main_all.py    --target_hardware flops   --gpu 0 '
#           '--warmup_epochs 0            --n_epochs 0              --retrain_epoch 1500 '
#           '--model_name Super_all       --mode retrain            --candidates_type single  --fast False --arch_lr 0.005 '
#           '--retrain_log_path   0,1_NUAA-SIRST_Super_all_single_17_10_2023_12_44_48')


#### Res18 channel two, 自适应channel 有shortcut Xavier初始化  remove self.net.init_model  3000 epoch
#### R15：74.4%  R16：74.6%  R17： 75.4%  R18：74.2%    AVG: 74.65%   (R19：75.6%)
# os.system('python  SIRST_main_all.py    --target_hardware flops   --gpu 0 '
#           '--warmup_epochs 0            --n_epochs 0              --retrain_epoch 3000 '
#           '--model_name Super_all       --mode retrain            --candidates_type single  --fast False --arch_lr 0.005 '
#           '--retrain_log_path   0,1_NUAA-SIRST_Super_all_single_17_10_2023_12_44_48')


# ### 12-07-18(Gene 2023_10_11_34 ResNet10 channel two)   Params: 692K  FLOPs: 3.31G
# R0: 73.1%  R2: 74.2%   R3: 74.5%   R4: 73.6%   AVG: 73.85%
# os.system('python  SIRST_main_all.py         --target_hardware flops   --gpu 1 '
#           '--warmup_epochs 0                 --n_epochs 0              --retrain_epoch 3000 '
#           '--model_name Super_all   --mode retrain            --candidates_type single  --fast False --arch_lr 0.005 '
#           '--retrain_log_path   0_NUAA-SIRST_Super_all_single_22_10_2023_12_07_18')


# ### 09_13_49(Gene 2023_10_11_34 ResNet18 channel two)   Params: 1.08M  FLOPs: 3.62G  比上面一个高是因为上面一个是resnet10
# ### R0: 73.2%  R2: 76.3%  R3: 75.0%  R4: 74.9%  Avg: 74.85%
# os.system('python  SIRST_main_all.py         --target_hardware flops   --gpu 0 '
#           '--warmup_epochs 0                 --n_epochs 0              --retrain_epoch 3000 '
#           '--model_name Super_all            --mode retrain            --candidates_type single  --fast False --arch_lr 0.005 '
#           '--retrain_log_path   0_NUAA-SIRST_Super_all_single_23_10_2023_09_13_49')


########################### 以Gene resnet_18_channel_two_layer_5/phase1_gene_Ushapeder_5_layer_down.txt，开展最终版前的初始测试
### ResConv    01-41-21  Search Flops: 2.1G-1.2G   72.6%  Retrain: 73.0%  Params: 0.91M
### GroupConv  23-39-58  Search Flops:0.52G-0.20G  68.3%  Retrain: 71.0%  Params: 0.11M
### SpaConv    02-40-13  Search Flops:1.37G-0.78G  72.1%  Retrain: 73.6%  Params: 0.62M
### DepthSC    06-10-24  Search Flops:0.52G-0.20G  68.3%  Retrain: 71.0%  Params: 0.06M
### MBInverted 04-53-10  Search Flops:0.94G-0.79G  63.6%  Retrain: 71.0%  Params: 0.33M

# os.system('python  SIRST_main_all.py    --target_hardware flops        --gpu 0,1 '
#           '--warmup_epochs 300          --n_epochs 1500                --retrain_epoch 1500 '
#           '--model_name Super_all       --mode all_search_train        --candidates_type ResConv  --fast False  --arch_lr 0.005 '
#           '--gene /media/gfkd/sda/NAS/NAS-DNANet-new-new/Result/search1/resnet_18_channel_two_layer_5/phase1_gene_Ushapeder_5_layer_down.txt ')


# ##测一下baseline Search 忽略，说明代码没问题  Retrain: 75.1%
# os.system('python  SIRST_main_all.py    --target_hardware flops        --gpu 0,1 '
#           '--warmup_epochs 0            --n_epochs 1                   --retrain_epoch 1500 '
#           '--model_name Super_all       --mode all_search_train        --candidates_type single   --fast False --arch_lr 0.005 '
#           '--gene /media/gfkd/sda/NAS/NAS-DNANet-new-new/Result/search1/resnet_18_channel_two_layer_5/phase1_gene_Ushapeder_5_layer_down.txt ')


########################### 以Gene 0_all_search_1_NUAA-SIRST_DNANet_22_09_2023_10_11_34/phase1_gene.txt，开展最终版侧事故
## 手动修改几个参数 把浅层改成参数量较小的GroupConv
## 初始：Params: 1.08M  FLOPs: 3.62G    改完如下：
#### human_final: layer=1全部为spa-conv  layer=2全部为groupconv，grup组数分别为2444
### R1: 74.0% R2: 74.3%  R3: 72.9%  R4: 73.9%  AVG: 73.7%    FLOPs: 2.16G  Params: 1.03M
# os.system('python  SIRST_main_all.py    --target_hardware flops   --gpu 0 '
#           '--warmup_epochs 0            --n_epochs 0              --retrain_epoch 1500 '
#           '--model_name Super_all       --mode retrain            --candidates_type single  --fast False --arch_lr 0.005 '
#           '--retrain_log_path   0_NUAA-SIRST_Super_all_single_23_10_2023_09_13_49_human_final')

# #### human1: layer=1全部为spa-conv
# # ### R1: 74.7%  R2: 73.8%  R3: 73.1%  R4: 74.0%  AVG: 73.9%  FLOPs: 2.67G  Params: 1.06M
# os.system('python  SIRST_main_all.py    --target_hardware flops   --gpu 1 '
#           '--warmup_epochs 0            --n_epochs 1              --retrain_epoch 1500 '
#           '--model_name Super_all       --mode retrain            --candidates_type single  --fast False --arch_lr 0.005 '
#           '--retrain_log_path   0_NUAA-SIRST_Super_all_single_23_10_2023_09_13_49_human_1 ')
#

# #### human3: 在human1的基礎上，layer=1全部为spa-conv， 大channel全部改为Group8,            FLOPs: 1.72G  Params: 0.767M
#### R0: 72.2%  R2: 73.3%  R3: 73.5%  R4: 74.7%  AVG: 73.42%
# os.system('python  SIRST_main_all.py    --target_hardware flops   --gpu 1 '
#           '--warmup_epochs 0            --n_epochs 1              --retrain_epoch 1500 '
#           '--model_name Super_all       --mode retrain            --candidates_type single  --fast False --arch_lr 0.005 '
#           '--retrain_log_path   0_NUAA-SIRST_Super_all_single_23_10_2023_09_13_49_human_3 ')


# #### human2: 在human1的基礎上，layer=1全部为spa-conv,且group爲2， 大channel全部改为Group8,  FLOPs: 1.35G  Params: 0.761M
#### R3: 74.6%   R4: 74.5%  R5: 74.3%  R6: 74.3%    AVG: 74.4%
# os.system('python  SIRST_main_all.py    --target_hardware flops   --gpu 0 '
#           '--warmup_epochs 0            --n_epochs 1              --retrain_epoch 1500 '
#           '--model_name Super_all       --mode retrain            --candidates_type single  --fast False --arch_lr 0.005 '
#           '--retrain_log_path   0_NUAA-SIRST_Super_all_single_23_10_2023_09_13_49_human_2 ')



# #### 01-09-52 Search忽略 Retrain: 74.6%  FLOPs: 3.26G Param: 0.92M
# os.system('python  SIRST_main_all.py    --target_hardware flops   --gpu 0,1 '
#           '--warmup_epochs 0            --n_epochs 1              --retrain_epoch 1500 '
#           '--model_name Super_all       --mode all_search_train   --candidates_type single  --fast False --arch_lr 0.005 '
#           '--gene  /media/gfkd/sda/NAS/NAS-DNANet-new-share-NEW/Result/search1/0_all_search_1_NUAA-SIRST_DNANet_23_10_2023_20_06_25/phase1_gene.txt ')
#
#
# #### 02-45-55 Search: FLOPs 14.8G-0.36G Params:0.06M  57.2% Retrain: 68.6%
# os.system('python  SIRST_main_all.py    --target_hardware flops   --gpu 0,1 '
#           '--warmup_epochs 300          --n_epochs 1500            --retrain_epoch 1500 '
#           '--model_name Super_all       --mode all_search_train   --candidates_type whole  --fast False --arch_lr 0.005 '
#           '--gene  /media/gfkd/sda/NAS/NAS-DNANet-new-share-NEW/Result/search1/0_all_search_1_NUAA-SIRST_DNANet_23_10_2023_20_06_25/phase1_gene.txt ')
#
# #### 07-38-38 Search: FLOPs 21.5G-0.41G Params:0.11M  64.7%  Retrain: 71.5%
# os.system('python  SIRST_main_all.py    --target_hardware flops   --gpu 0,1 '
#           '--warmup_epochs 300          --n_epochs 1500            --retrain_epoch 1500 '
#           '--model_name Super_all       --mode all_search_train   --candidates_type Res_Group_Spa_MBConv  --fast False --arch_lr 0.005 '
#           '--gene  /media/gfkd/sda/NAS/NAS-DNANet-new-share-NEW/Result/search1/0_all_search_1_NUAA-SIRST_DNANet_23_10_2023_20_06_25/phase1_gene.txt ')
#
# #### 12-47-26 Seach:  Cpu: 實際：407-98ms 理論：290-59.5ms 64.6%  Retrain: 75.0% FLOPs:2.96G  Params: 0.23M
# os.system('python  SIRST_main_all.py    --target_hardware cpu     --gpu 0,1 '
#           '--warmup_epochs 300          --n_epochs 1500           --retrain_epoch 1500 '
#           '--model_name Super_all       --mode all_search_train   --candidates_type Res_Group_Spa_MBConv  --fast False --arch_lr 0.005 '
#           '--gene  /media/gfkd/sda/NAS/NAS-DNANet-new-share-NEW/Result/search1/0_all_search_1_NUAA-SIRST_DNANet_23_10_2023_20_06_25/phase1_gene.txt ')


### 12-47-26的多次Retrain结果，
# R2: 72.8%   R3: 72.8%  R4: 74.7%  R5: 73.4% AVG: 73.42%
# R6: 74.1%   R7: 73.6%  R8: 73.5%  R9: 74.7% R10: 72.4% R11: 74.9% R12: 74.7%
# os.system('python  SIRST_main_all.py    --target_hardware gpu     --gpu 1 '
#               '--warmup_epochs 0            --n_epochs 1              --retrain_epoch 1500 '
#               '--model_name Super_all       --mode retrain            --candidates_type Res_Group_Spa_MBConv  --fast False --arch_lr 0.005 '
#               '--retrain_log_path  0,1_NUAA-SIRST_Super_all_Res_Group_Spa_MBConv_26_10_2023_12_47_26 ')


### 10-47-00 GPU: 实际：6.2ms- 5.1ms 理论：14.8ms-6.2ms 66.0%
# os.system('python  SIRST_main_all.py    --target_hardware gpu     --gpu 0,1 '
#           '--warmup_epochs 300          --n_epochs 1500           --retrain_epoch 1500 '
#           '--model_name Super_all       --mode all_search_train   --candidates_type Res_Group_Spa_MBConv  --fast False --arch_lr 0.005 '
#           '--gene  /media/gfkd/sda/NAS/NAS-DNANet-new-share-NEW/Result/search1/0_all_search_1_NUAA-SIRST_DNANet_23_10_2023_20_06_25/phase1_gene.txt ')

### 10-47-00 Retrain: 73.7%  R2: 74.4% R3: 71.1% R4: 74.8%
# os.system('python  SIRST_main_all.py    --target_hardware gpu     --gpu 0 '
#           '--warmup_epochs 0            --n_epochs 1              --retrain_epoch 1500 '
#           '--model_name Super_all       --mode retrain            --candidates_type Res_Group_Spa_MBConv  --fast False --arch_lr 0.005 '
#           '--gene /media/gfkd/sda/NAS/NAS-DNANet-new-share-NEW/Result/search1/0_all_search_1_NUAA-SIRST_DNANet_23_10_2023_20_06_25/phase1_gene.txt '
#           '--retrain_log_path  0,1_NUAA-SIRST_Super_all_Res_Group_Spa_MBConv_27_10_2023_10_47_00 ')



#### 降低arch loss的权重， 没有group8 *10 latency
##### 01-23-52 Flops:4.1G->0.4G 44.3%  Retrain: 72.3%  mul#log  grad_reg_loss_alpha 0.2       grad_reg_loss_beta 0.3
##### 04-16-17 Flops:4.1G->0.4G 48.5%  Retrain: 68.9%  mul#log  grad_reg_loss_alpha 0.1       grad_reg_loss_beta 0.1
##### 07-12-47 Flops:2.1G->0.4G 37.3%  Retrain: 69.1%  mul#log  grad_reg_loss_alpha 0.01      grad_reg_loss_beta 0.01
##### 10-10-29 Flops:3.9G->0.4G 19.8%  Retrain: 71.0%  mul#log  grad_reg_loss_type add#linear grad_reg_loss_lambda 0.1
##以上方法没用，因为都会落到一个searched的网络上
# os.system('python  SIRST_main_all.py    --target_hardware flops   --gpu 0,1 '
#           '--warmup_epochs 300          --n_epochs 500            --retrain_epoch 1500 '
#           '--model_name Super_all       --mode all_search_train    --candidates_type Res_Group_Spa_MBConv  --fast False --arch_lr 0.005 '
#           '--grad_reg_loss_type mul#log --grad_reg_loss_alpha 0.2 --grad_reg_loss_beta 0.3 '
#           '--gene /media/gfkd/sda/NAS/NAS-DNANet-new-share-NEW/Result/search1/0_all_search_1_NUAA-SIRST_DNANet_23_10_2023_20_06_25/phase1_gene.txt '
#           '--retrain_log_path  0,1_NUAA-SIRST_Super_all_Res_Group_Spa_MBConv_27_10_2023_10_47_00 ')



#### 换成add#linear_abs
##### 17-21-51 Search  FLops卡在10G 44.5%                       Retrain: 74.4% --ref_value 10000000000
##### 00-10-30 Search  Flops: 0.4G (加权理论值收敛到3G了)， 25.2%  Retrain: 67.9% --ref_value 3000000000

# os.system('python  SIRST_main_all.py    --target_hardware flops   --gpu 0,1 '
#           '--warmup_epochs 300          --n_epochs 500            --retrain_epoch 1500 '
#           '--model_name Super_all       --mode all_search_train    --candidates_type Res_Group_Spa_MBConv  --fast False --arch_lr 0.005 '
#           '--grad_reg_loss_type add#linear_abs --ref_value 10000000000 '
#           '--gene /media/gfkd/sda/NAS/NAS-DNANet-new-share-NEW/Result/search1/0_all_search_1_NUAA-SIRST_DNANet_23_10_2023_20_06_25/phase1_gene.txt '
#           '--retrain_log_path  0,1_NUAA-SIRST_Super_all_Res_Group_Spa_MBConv_27_10_2023_10_47_00 ')


##### 09-25-15 Search: 4.1G-0.41G  ref_value 5000000000
##### 09-28-44 Search: 4.1G-0.46G  ref_value 6000000000
##### 09-31-57 Search: 3.3G-0.68G  ref_value 7000000000
##### 12-16-42 Search: 2.7G-1.20G  ref_value 7100000000 Retrain: R1:68.4%  R2:69.3%  R3:69.1%
##### 16-08-33 Search: 9.3G-1.09G  ref_value 7200000000 Retrain: R1:69.1%
##### 16-20-47 Search: 18.0G-0.72G ref_value 7300000000
##### 16-33-16 Search: 2.9G-2.7G   ref_value 7400000000
##### 16-45-43 Search: 8.2G-5.0G   ref_value 7500000000
##### 16-58-23 Search: 8.9G-4.7G   ref_value 7600000000
##### 17-10-53 Search: 1.2G-4.7G   ref_value 7700000000
##### 17-23-38 Search: 3.9G-5.3G   ref_value 7800000000
##### 17-36-46 Search: 1.2G-4.7G   ref_value 7900000000
##### 09-44-29 Search: 15G-6.4G    ref_value 8000000000
##### 09-58-05 Search: 8.2G-8.9G   ref_value 9000000000

# os.system('python  SIRST_main_all.py    --target_hardware flops   --gpu 0,1 '
#           '--warmup_epochs 0            --n_epochs 100              --retrain_epoch 0 '
#           '--model_name Super_all       --mode all_search_train    --candidates_type Res_Group_Spa_MBConv  --fast False --arch_lr 0.005 '
#           '--grad_reg_loss_type add#linear_abs --ref_value 5000000000 '
#           '--gene /media/gfkd/sda/NAS/NAS-DNANet-new-share-NEW/Result/search1/0_all_search_1_NUAA-SIRST_DNANet_23_10_2023_20_06_25/phase1_gene.txt ')


##### 21-49-29 Search: FLOPs: 2.8G-0.74G   Retrain: 71.7%
##### 00-10-48 Search: FLOPs: 9.7G-0.7G    Retrain: 70.5%
##### 02-36-12 Search: FLOPs: 2.7G-0.5G    Retrain: 71.3%
##### 05-06-03 Search: FLOPs: 14.4G-0.8G   Retrain: 68.7%
##### 07-42-53 Search: FLOPs: 3.5G-0.8G    Retrain: 71.7%

# os.system('python  SIRST_main_all.py       --target_hardware flops    --gpu 0,1 '
#           '--warmup_epochs 300             --n_epochs 200             --retrain_epoch 1500 '
#           '--model_name Super_all          --mode all_search_train    --candidates_type Res_Group_Spa_MBConv  --fast False --arch_lr 0.005 '
#           '--grad_reg_loss_type add#linear --random_choose True '
#           '--gene /media/gfkd/sda/NAS/NAS-DNANet-new-share-NEW/Result/search1/0_all_search_1_NUAA-SIRST_DNANet_23_10_2023_20_06_25/phase1_gene.txt ')


##### 11-26-35 Search: Flops: 4.2G->2.1G
##### 11-34-58 Search: Flops: 2.2G->1.3G
##### 11-44-37 Search: Flops: 1.7G->1.1G
##### 11-53-30 Search: Flops: 5.5G->1.6G

# os.system('python  SIRST_main_all.py       --target_hardware flops    --gpu 0,1 '
#           '--warmup_epochs 0               --n_epochs 100             --retrain_epoch 0 '
#           '--model_name Super_all          --mode all_search_train    --candidates_type Res_Group_Spa_MBConv  --fast False --arch_lr 0.005 '
#           '--grad_reg_loss_type add#linear --random_choose True '
#           '--gene /media/gfkd/sda/NAS/NAS-DNANet-new-share-NEW/Result/search1/0_all_search_1_NUAA-SIRST_DNANet_23_10_2023_20_06_25/phase1_gene.txt ')

# ##### 11-26-35 Retrain: 73.2%  73.3%  FLOPs: 2.1G
# os.system('python  SIRST_main_all.py    --target_hardware gpu     --gpu 0 '
#           '--warmup_epochs 0            --n_epochs 1              --retrain_epoch 1500 '
#           '--model_name Super_all       --mode retrain            --candidates_type Res_Group_Spa_MBConv  --fast False --arch_lr 0.005 '
#           '--gene /media/gfkd/sda/NAS/NAS-DNANet-new-share-NEW/Result/search1/0_all_search_1_NUAA-SIRST_DNANet_23_10_2023_20_06_25/phase1_gene.txt '
#           '--retrain_log_path  0,1_NUAA-SIRST_Super_all_Res_Group_Spa_MBConv_01_11_2023_11_26_35 ')
#
# ##### 11-34-58 Retrain: 72.6%  74.6%  FLOPs: 1.3G
# os.system('python  SIRST_main_all.py    --target_hardware gpu     --gpu 1 '
#           '--warmup_epochs 0            --n_epochs 1              --retrain_epoch 1500 '
#           '--model_name Super_all       --mode retrain            --candidates_type Res_Group_Spa_MBConv  --fast False --arch_lr 0.005 '
#           '--gene /media/gfkd/sda/NAS/NAS-DNANet-new-share-NEW/Result/search1/0_all_search_1_NUAA-SIRST_DNANet_23_10_2023_20_06_25/phase1_gene.txt '
#           '--retrain_log_path  0,1_NUAA-SIRST_Super_all_Res_Group_Spa_MBConv_01_11_2023_11_34_58 ')



##### 看哪一个子operation效果最好， AutoDL上跑
#### 00-24-19 Retrain: 65.8%  FLOPs: 0.48G   Params: 0.06M  3x3_DepthSC
#### 02-34-56 Retrain: 70.0%  FLOPs: 0.87G   Params: 0.08M  5x5_DepthSC
#### 04-59-54 Retrain: 71.1%  FLOPs: 1.46G   Params: 0.11M  7x7_DepthSC
#### 07-44-10 Retrain: 69.7%  FLOPs: 9.57G   Params: 0.42M  3x3_MBConv2
#### 11-49-18 Retrain: 70.4%  FLOPs: 19.1G   Params: 0.83M  3x3_MBConv4
#### xx-xx-xx Retrain: 69.7%  FLOPs: 9.57G   Params: 0.42M  3x3_MBConv6
#### 11-18-43 Retrain: 74.7%  FLOPs: 3.26G   Params: 0.92M  3x3_ResConv
# Retrain:    19-57-11 75.7%  22-02-34 73.3% 00-11-32 73.7% 02-21-51 72.0%  04-28-23 75.2%
###  13-49-42 Retrain: 75.1%  FLOPs: 8.62G   Params: 2.50M  5x5_ResConv
# Retrain:    06-36-46 73.2%  09-27-32 73.6% 12-23-11 74.8% 15-15-51 73.5%  18-08-20 74.7%
###  17-37-55 Retrain: 72.8%  FLOPs: 16.6G   Params: 4.87M  7x7_ResConv
# Retrain:    21-03-55 72.1%  01-57-26 72.8% 06-50-25 73.6% 11-46-49 72.5%  16-41-43 73.0%

#### 23-51-46 Retrain: 74.3%  FLOPs: 1.65G   Params: 0.46M  3x3_GroupConv2
#### 04-32-31 Retrain: 73.5%  FLOPs: 0.85G   Params: 0.23M  3x3_GroupConv4
#### 09-05-46 Retrain: 68.5%  FLOPs: 0.45G   Params: 0.12M  3x3_GroupConv8
#### 13-13-04 Retrain: 73.8%  FLOPs: 1.65G   Params: 0.62M  3x3_SpaConv
#### 15-31-32 Retrain: 72.3%  FLOPs: 2.59G   Params: 1.01M  5x5_SpaConv
#### 18-03-54 Retrain: 74.3%  FLOPs: 3.53G   Params: 1.40M  7x7_SpaConv

# os.system('python  SIRST_main_all.py    --target_hardware flops        --gpu 0 '
#           '--warmup_epochs 0            --n_epochs 1                   --retrain_epoch 1500 '
#           '--model_name Super_all       --mode all_search_train        --candidates_type 3x3_DepthSC   --fast False --arch_lr 0.005 '
#           '--gene /media/gfkd/sda/NAS/NAS-DNANet-new-share-NEW/Result/search1/0_all_search_1_NUAA-SIRST_DNANet_23_10_2023_20_06_25/phase1_gene.txt ')




# ###### 本pc上跑, group8 *10 latency
# #### 00-29-03 Search-flops: 1.9G-0.7G  59.9%  Retrain: 67.4%
# os.system('python  SIRST_main_all.py    --target_hardware flops    --gpu 0,1 '
#           '--warmup_epochs 300          --n_epochs 1500            --retrain_epoch 1500 '
#           '--model_name Super_all       --mode all_search_train    --candidates_type Res_Group_Spa_MBConv  --fast False --arch_lr 0.005 '
#           '--gene /media/gfkd/sda/NAS/NAS-DNANet-new-share-NEW/Result/search1/0_all_search_1_NUAA-SIRST_DNANet_23_10_2023_20_06_25/phase1_gene.txt ')
#
# #### 05-26-18 Search-flops: 8.4G-0.7G  61.7%  Retrain: 69.7%
# os.system('python  SIRST_main_all.py    --target_hardware flops    --gpu 0,1 '
#           '--warmup_epochs 300          --n_epochs 1500            --retrain_epoch 1500 '
#           '--model_name Super_all       --mode all_search_train    --candidates_type Res_Group_Spa_MBConv  --fast False --arch_lr 0.005 '
#           '--gene /media/gfkd/sda/NAS/NAS-DNANet-new-share-NEW/Result/search1/0_all_search_1_NUAA-SIRST_DNANet_23_10_2023_20_06_25/phase1_gene.txt ')


#### 21-47-10  Search: 7.6G-3.2G   Retrain: 75.1%
#### 02-06-42  Search: 11.7G-3.2G  Retrain: 75.8%
# os.system('python  SIRST_main_all.py       --target_hardware flops    --gpu 0,1 '
#           '--warmup_epochs 1500            --n_epochs 500             --retrain_epoch 1500 '
#           '--model_name Super_all          --mode all_search_train    --candidates_type ResConv  --fast False --arch_lr 0.005 '
#           '--grad_reg_loss_type add#linear --random_choose False '
#           '--gene /media/gfkd/sda/NAS/NAS-DNANet-new-share-NEW/Result/search1/0_all_search_1_NUAA-SIRST_DNANet_23_10_2023_20_06_25/phase1_gene.txt ')


#### 06-31-40  Search: 5.6G-2.2G  Retrain: 73.7%
#### 13-25-36  Search: 6.9G-2.2G  Retrain: 75.4%
# os.system('python  SIRST_main_all.py       --target_hardware flops    --gpu 0,1 '
#           '--warmup_epochs 1500            --n_epochs 500             --retrain_epoch 1500 '
#           '--model_name Super_all          --mode all_search_train    --candidates_type Res_Group  --fast False --arch_lr 0.005 '
#           '--grad_reg_loss_type add#linear --random_choose False '
#           '--gene /media/gfkd/sda/NAS/NAS-DNANet-new-share-NEW/Result/search1/0_all_search_1_NUAA-SIRST_DNANet_23_10_2023_20_06_25/phase1_gene.txt ')



# # #### 17-20-16  Search: 理论：2984->2984不动，实际：227->351ms, Retrain: 71.8%  one-hot cpu latency代码有问题，显示的理论latency不对，已经改过来了
# # #### 21-56-27  Search: 理论：2984->2984不动，实际：172->332ms, Retrain: 72.9%  one-hot cpu latency代码有问题，显示的理论latency不对，已经改过来了
# ###### 05-56-13  Search: 理论cpu:2956-1334ms  实际cpu: 161.5-108.7ms  Retrain: 72.5%
# os.system('python  SIRST_main_all.py       --target_hardware cpu      --gpu 0,1 '
#           '--warmup_epochs 1500            --n_epochs 100             --retrain_epoch 1500 '
#           '--model_name Super_all          --mode all_search_train    --candidates_type Res_Group_Spa  --fast False --arch_lr 0.005 '
#           '--grad_reg_loss_type add#linear --random_choose False '
#           '--gene /media/gfkd/sda/NAS/NAS-DNANet-new-share-NEW/Result/search1/0_all_search_1_NUAA-SIRST_DNANet_23_10_2023_20_06_25/phase1_gene.txt ')



# # #### 02-34-29  Search: 理论：145.2->145.2，实际：6.3->5.5ms, Retrain: 73.4%  one-hot gpu latency代码有问题，显示的理论latency不对，已经改过来了
# # #### 07-14-55  Search: 理论：145.2->145.2，实际：6.0->5.5ms, Retrain: 73.2%  one-hot gpu latency代码有问题，显示的理论latency不对，已经改过来了
# ###### 10-23-50  Search: 理论：11.7->6.8，   实际：6.2->5.5ms, Retrain: 73.9%
# os.system('python  SIRST_main_all.py       --target_hardware gpu      --gpu 0,1 '
#           '--warmup_epochs 1500            --n_epochs 100             --retrain_epoch 1500 '
#           '--model_name Super_all          --mode all_search_train    --candidates_type Res_Group_Spa_MBConv  --fast False --arch_lr 0.005 '
#           '--grad_reg_loss_type add#linear --random_choose False '
#           '--gene /media/gfkd/sda/NAS/NAS-DNANet-new-share-NEW/Result/search1/0_all_search_1_NUAA-SIRST_DNANet_23_10_2023_20_06_25/phase1_gene.txt ')


# #### 19-33-42  Search: 时间没显示  Retrain: 73.4%
# #### 00-20-25  Searcg: 理论：90.23-60.05ms  Retrain：73.5%
# os.system('python  SIRST_main_all.py       --target_hardware edge-gpu   --gpu 0,1 '
#           '--warmup_epochs 1500            --n_epochs 100               --retrain_epoch 1500 '
#           '--model_name Super_all          --mode all_search_train      --candidates_type Res_Group_Spa_MBConv  --fast False --arch_lr 0.005 '
#           '--grad_reg_loss_type add#linear --random_choose False '
#           '--gene /media/gfkd/sda/NAS/NAS-DNANet-new-share-NEW/Result/search1/0_all_search_1_NUAA-SIRST_DNANet_23_10_2023_20_06_25/phase1_gene.txt ')
#
# #### 00-37-52  Search: 时间没显示  Retrain: 74.9%
# #### 05-14-06  Searcg: 理论：4448-1048ms  Retrain：74.5%
# os.system('python  SIRST_main_all.py       --target_hardware edge-cpu   --gpu 0,1 '
#           '--warmup_epochs 1500            --n_epochs 100               --retrain_epoch 1500 '
#           '--model_name Super_all          --mode all_search_train      --candidates_type Res_Group_Spa_MBConv  --fast False --arch_lr 0.005 '
#           '--grad_reg_loss_type add#linear --random_choose False '
#           '--gene /media/gfkd/sda/NAS/NAS-DNANet-new-share-NEW/Result/search1/0_all_search_1_NUAA-SIRST_DNANet_23_10_2023_20_06_25/phase1_gene.txt ')
#
# ##### 05-56-13 Retrain结果：
# #### R2：72.6%  R3：73.3%  R4：74.4%  R5：72.7%  R6：74.1%
# for i in range(5):
#     os.system('python  SIRST_main_all.py       --target_hardware cpu      --gpu 0,1 '
#               '--warmup_epochs 0               --n_epochs 0               --retrain_epoch 1500 '
#               '--model_name Super_all          --mode retrain             --candidates_type Res_Group_Spa  --fast False --arch_lr 0.005 '
#               '--grad_reg_loss_type add#linear --random_choose False '
#               '--gene /media/gfkd/sda/NAS/NAS-DNANet-new-share-NEW/Result/search1/0_all_search_1_NUAA-SIRST_DNANet_23_10_2023_20_06_25/phase1_gene.txt '
#               '--retrain_log_path 0,1_NUAA-SIRST_Super_all_Res_Group_Spa_13_11_2023_05_56_13 ')
#
# ###### 10-23-50 Retrain结果:
# #### R2：72.8%  R3：73.1%  R4：72.8%  R5：70.9%  R6：73.3%
# for i in range(5):
#     os.system('python  SIRST_main_all.py       --target_hardware cpu      --gpu 0,1 '
#               '--warmup_epochs 0               --n_epochs 0               --retrain_epoch 1500 '
#               '--model_name Super_all          --mode retrain             --candidates_type Res_Group_Spa  --fast False --arch_lr 0.005 '
#               '--grad_reg_loss_type add#linear --random_choose False '
#               '--gene /media/gfkd/sda/NAS/NAS-DNANet-new-share-NEW/Result/search1/0_all_search_1_NUAA-SIRST_DNANet_23_10_2023_20_06_25/phase1_gene.txt '
#               '--retrain_log_path 0,1_NUAA-SIRST_Super_all_Res_Group_Spa_MBConv_13_11_2023_10_23_50 ')


# #### 重新训一下这个
# #### FLOPs NUDT-SIRST
# #### 11-34-58 R7 81.0%
# #### 11-34-58 R4 80.1%
# os.system('python  SIRST_main_all.py       --target_hardware cpu      --gpu 0 '
#           '--warmup_epochs 0               --n_epochs 0               --retrain_epoch 1500 '
#           '--model_name Super_all          --mode retrain             --dataset NUDT-SIRST    --split_method 50_50   '
#           '--gene /media/gfkd/sda/NAS/NAS-DNANet-new-share-NEW/Result/search1/0_all_search_1_NUAA-SIRST_DNANet_23_10_2023_20_06_25/phase1_gene.txt '
#           '--retrain_log_path  0,1_NUAA-SIRST_Super_all_Res_Group_Spa_MBConv_01_11_2023_11_34_58 ')


### Proxyless
### 10-33-21 Retrain: 69.3%  93.15%  31.23x10(-6)
# os.system('python  SIRST_main_all.py       --target_hardware flops    --gpu 0,1 '
#           '--warmup_epochs 300             --n_epochs 100             --retrain_epoch 1500 '
#           '--model_name Super_all          --mode all_search_train    --dataset NUAA-SIRST    --split_method 50_50 '
#           '--candidates_type Proxyless     --fast False               --arch_lr 0.005 '
#           '--grad_reg_loss_type add#linear --random_choose False      --train_batch_size 8    --test_batch_size 8 '
#           '--gene /media/gfkd/sda/NAS/NAS-DNANet-new-share-NEW/Result/search1/0_all_search_1_NUAA-SIRST_DNANet_23_10_2023_20_06_25/phase1_gene.txt ')

### 10-33-21 R4： 75.4%  95.34%  27.04*10(-6)
# os.system('python  SIRST_main_all.py       --target_hardware flops    --gpu 0 '
#           '--warmup_epochs 0               --n_epochs 0               --retrain_epoch 1500 '
#           '--model_name Super_all          --mode retrain             --dataset NUDT-SIRST    --split_method 50_50   '
#           '--gene /media/gfkd/sda/NAS/NAS-DNANet-new-share-NEW/Result/search1/0_all_search_1_NUAA-SIRST_DNANet_23_10_2023_20_06_25/phase1_gene.txt '
#           '--retrain_log_path  0,1_NUAA-SIRST_Super_all_Proxyless_01_12_2023_10_33_21 ')

## Proxyless Retrain on IRSTD-SIRST   R2:  67.0%  87.75% 7.06x10(-6)
os.system('python  SIRST_main_all.py       --target_hardware flops    --gpu 0 '
          '--warmup_epochs 0               --n_epochs 0               --retrain_epoch 1500 '
          '--model_name Super_all          --mode retrain             --dataset IRSTD-SIRST    --split_method 80_20   '
          '--gene /media/gfkd/sda/NAS/NAS-DNANet-new-share-NEW/Result/search1/0_all_search_1_NUAA-SIRST_DNANet_23_10_2023_20_06_25/phase1_gene.txt '
          '--retrain_log_path  0,1_NUAA-SIRST_Super_all_Proxyless_01_12_2023_10_33_21 ')

#### Meta 20-05-09 R2: 74.8% 96.57(%)  16.54(-6)
os.system('python  SIRST_main_all.py       --target_hardware flops    --gpu 0 '
          '--warmup_epochs 0               --n_epochs 0               --retrain_epoch 1500 '
          '--model_name Super_all          --mode retrain             --dataset NUAA-SIRST    --split_method 50_50 '
          '--gene /media/gfkd/sda/NAS/NAS-DNANet-new-share-NEW/Result/search1/0_all_search_1_NUAA-SIRST_DNANet_Meta/phase1_gene.txt '
          '--retrain_log_path  0_NUAA-SIRST_Super_all_single_06_01_2024_20_05_09 ')


# # #### Central IRSTD-SIRST  AUTODL上训练
# #### 12-47-26 R13     R14     R15
# #             64.6%  63.2%  67.0%
# os.system('python  SIRST_main_all.py       --target_hardware cpu      --gpu 0 '
#           '--warmup_epochs 0               --n_epochs 0               --retrain_epoch 1500 '
#           '--model_name Super_all          --mode retrain             --dataset IRSTD-SIRST    --split_method 80_20   '
#           '--gene /media/gfkd/sda/NAS/NAS-DNANet-new-share-NEW/Result/search1/0_all_search_1_NUAA-SIRST_DNANet_23_10_2023_20_06_25/phase1_gene.txt '
#           '--retrain_log_path  0,1_NUAA-SIRST_Super_all_Res_Group_Spa_MBConv_26_10_2023_12_47_26 ')
# #### 10-47-00   R5    R6     R7
# #              64.6%  67.1%  68.2%
# os.system('python  SIRST_main_all.py       --target_hardware cpu      --gpu 0 '
#           '--warmup_epochs 0               --n_epochs 0               --retrain_epoch 1500 '
#           '--model_name Super_all          --mode retrain             --dataset IRSTD-SIRST    --split_method 80_20   '
#           '--gene /media/gfkd/sda/NAS/NAS-DNANet-new-share-NEW/Result/search1/0_all_search_1_NUAA-SIRST_DNANet_23_10_2023_20_06_25/phase1_gene.txt '
#           '--retrain_log_path  0,1_NUAA-SIRST_Super_all_Res_Group_Spa_MBConv_27_10_2023_10_47_00 ')
