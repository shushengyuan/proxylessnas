import os

### 23-30-36 train 100个epoch 结果忽略；    retrain 71.2%  学到的conv只有3x3_MBConv3，候选是所有MBConv。

### 11-48-45 warm 200, train 1500, cpu 104-76ms   63.3%； 没有 retrain

### 16-36-00 warm 300. train 500,  gpu 5.7-4.6ms  58.6%;  没有 retrain

### 15-37-47 train 结果忽略；              retrain 65.3%  ， 候选conv只有3x3_MBConv1

### 22-35-46 train 结果忽略；              retrain 73.6%  ， 候选conv只有3x3_ResConv0

### 17-23-13 train 结果忽略；R 2, retrain 74.3%     候选只有3x3_ResConv  retrain-2 73.7%

### 21-57-06 warm 300,  train 1500, gpu16-11ms      39.9%; retrain 候选所有res 和MB， retrain结果没有，后面可以再试一下

### 15-38-47 warm 100,  train 100,  gpu58-55ms R-1  29.4%; retrain 候选所有res 和MB,  retrain 72.1%

### 00-09-32 warm 300,  train 1500, cpu148-59ms     49.2%; retrain 候选完全体，       retrain 48.7% 【half】

### 11-22-49 train 结果忽略；                                retrain 候选3x3_ResConv， retrain 66.7% 【half】

# ### 22-35-46-Retrain  retrain(覆盖了上面的retrain) 73.2%
# os.system('python  SIRST_main_all.py  --mode retrain --model_name Super_all --retrain_log_path 0,1_NUAA-SIRST_Super_28_08_2023_22_35_46')

# ### 17-23-13-Retrain3 retrain 73.9%
# os.system('python  SIRST_main_all.py  --mode retrain --model_name Super_all --retrain_log_path 0,1_NUAA-SIRST_Super_29_08_2023_17_23_13')


# ### 00-28-12 train 结果忽略； retrain 完全体 72.5%
# os.system('python  SIRST_main_all.py --target_hardware cpu   --gpu 0,1 --mode all_search_train --model_name Super_all')
#
# ### 02-22-49 train 结果忽略； retrain 完全体 73.9%
# os.system('python  SIRST_main_all.py --target_hardware gpu   --gpu 0,1 --mode all_search_train --model_name Super_all')


# ### 23-01-05 Retrain   74.0%
# os.system('python  SIRST_main_all.py  --gpu 0 --mode retrain --model_name Super_all --retrain_log_path 0,1_NUAA-SIRST_Super_all_04_09_2023_23_01_05')
#
# ### 23-01-05 Retrain_2 71.4%
# os.system('python  SIRST_main_all.py  --gpu 1 --mode retrain --model_name Super_all --retrain_log_path 0,1_NUAA-SIRST_Super_all_04_09_2023_23_01_05')


####### 23-20-25 train 46.1%   Retrain 25.5%  代码错了，搞了9次下采样
###0. Mix(3x3_ResConv, 1.000)
###1. Mix(7x7_MBConv1, 0.500)
###2. Mix(5x5_MBConv1, 0.431)
###3. Mix(7x7_MBConv1, 0.442)
###4. Mix(7x7_MBConv1, 0.523)
###5. Mix(7x7_MBConv1, 0.442)
###6. Mix(5x5_MBConv1, 0.431)
###7. Mix(7x7_MBConv1, 0.500)
###8. Mix(3x3_ResConv, 1.000)
# os.system('python  SIRST_main_all.py --target_hardware gpu   --gpu 0,1 --warmup_epochs 300  --n_epochs 1500 --retrain_epoch 1500 --mode all_search_train
#           --model_name Super_half')
#
#
# ####### 01-59-31 train 50.2%  Retrain 54.4%  代码错了，搞了9次下采样
# ###0. Mix(3x3_ResConv, 1.000)
# ###1. Mix(3x3_ResConv, 0.179)
# ###2. Mix(5x5_MBConv1, 0.200)
# ###3. Mix(5x5_MBConv1, 0.316)
# ###4. Mix(3x3_ResConv, 0.486)
# ###5. Mix(5x5_MBConv1, 0.316)
# ###6. Mix(5x5_MBConv1, 0.200)
# ###7. Mix(3x3_ResConv, 0.179)
# ###8. Mix(3x3_ResConv, 1.000)
# os.system('python  SIRST_main_all.py --target_hardware cpu   --gpu 0,1 --warmup_epochs 300  --n_epochs 1500 --retrain_epoch 1500 --mode all_search_train
#           --model_name Super_half')
#
#
# ####### 05-05-30 train 49.5%  Retrain 32.3%  代码错了，搞了9次下采样
# ###0. Mix(3x3_ResConv, 1.000)
# ###1. Mix(3x3_MBConv1, 0.133)
# ###2. Mix(3x3_MBConv1, 0.131)
# ###3. Mix(7x7_MBConv1, 0.116)
# ###4. Mix(5x5_MBConv1, 0.089)
# ###5. Mix(7x7_MBConv1, 0.116)
# ###6. Mix(3x3_MBConv1, 0.131)
# ###7. Mix(3x3_MBConv1, 0.133)
# ###8. Mix(3x3_ResConv, 1.000)
# os.system('python  SIRST_main_all.py --target_hardware flops --gpu 0,1 --warmup_epochs 300  --n_epochs 1500 --retrain_epoch 1500 --mode all_search_train
#           --model_name Super_half')


# ####### 11-32-24 train 57.4%  Retrain 70.4%  代码错了，搞了9次下采样
# ###0. Mix(3x3_ResConv, 1.000)
# ###1. Mix(7x7_ResConv, 0.965)
# ###2. Mix(5x5_ResConv, 0.961)
# ###3. Mix(5x5_ResConv, 0.962)
# ###4. Mix(7x7_ResConv, 0.514)
# ###5. Mix(5x5_ResConv, 0.962)
# ###6. Mix(5x5_ResConv, 0.961)
# ###7. Mix(7x7_ResConv, 0.965)
# ###8. Mix(3x3_ResConv, 1.000)
# os.system('python  SIRST_main_all.py --target_hardware gpu   --gpu 0,1 --warmup_epochs 300  --n_epochs 1500 --retrain_epoch 1500 --mode all_search_train
#           --model_name Super_half --candidates_type ResConv')


os.system('python  SIRST_main_all.py --target_hardware gpu    --gpu 0,1 --warmup_epochs 300  --n_epochs 1500 --retrain_epoch 1500 --mode all_search_train --model_name Super_half --candidates_type ResConv')
os.system('python  SIRST_main_all.py --target_hardware cpu    --gpu 0,1 --warmup_epochs 300  --n_epochs 1500 --retrain_epoch 1500 --mode all_search_train --model_name Super_half --candidates_type all ')
os.system('python  SIRST_main_all.py --target_hardware gpu    --gpu 0,1 --warmup_epochs 300  --n_epochs 1500 --retrain_epoch 1500 --mode all_search_train --model_name Super_half --candidates_type all ')
os.system('python  SIRST_main_all.py --target_hardware flops  --gpu 0,1 --warmup_epochs 300  --n_epochs 1500 --retrain_epoch 1500 --mode all_search_train --model_name Super_half --candidates_type all ')




