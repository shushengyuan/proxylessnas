import os

# ### 00-28-15 train 可以忽略； retrain 73.4%
# os.system('python  SIRST_main_all.py --target_hardware cpu   --gpu 0,1 --mode all_search_train --model_name Super_all')
#
# ### 02-28-33 train 可以忽略； retrain 73.0%
# os.system('python  SIRST_main_all.py --target_hardware gpu   --gpu 0,1 --mode all_search_train --model_name Super_all')



####### 23-20-27 train 44.2%  Retrain 72.9%
###0. Mix(3x3_ResConv, 1.000)
###1. Mix(7x7_MBConv1, 0.507)
###2. Mix(7x7_MBConv1, 0.425)
###3. Mix(7x7_MBConv1, 0.473)
###4. Mix(7x7_MBConv1, 0.514)
###5. Mix(3x3_MBConv1, 0.789)
###6. Mix(7x7_ResConv, 0.402)
###7. Mix(7x7_MBConv1, 0.825)
###8. Mix(7x7_MBConv1, 0.481)
# os.system('python  SIRST_main_all.py --target_hardware gpu   --gpu 1,0 --warmup_epochs 300  --n_epochs 1500 --retrain_epoch 1500
#           --mode all_search_train --model_name Super_all')
#
# ####### 01-50-32 train 49.3% Retrain 73.6%
# ###0. Mix(3x3_ResConv, 1.000)
# ###1. Mix(3x3_ResConv, 0.172)
# ###2. Mix(3x3_ResConv, 0.171)
# ###3. Mix(7x7_MBConv1, 0.363)
# ###4. Mix(3x3_ResConv, 0.523)
# ###5. Mix(7x7_MBConv1, 0.297)
# ###6. Mix(5x5_MBConv1, 0.189)
# ###7. Mix(3x3_ResConv, 0.164)
# ###8. Mix(3x3_ResConv, 0.161)
# os.system('python  SIRST_main_all.py --target_hardware cpu   --gpu 1,0 --warmup_epochs 300  --n_epochs 1500 --retrain_epoch 1500
#           --mode all_search_train --model_name Super_all')
#
# ####### 04-57-08 train 42.6% Retrain 69.2%
# ###0. Mix(3x3_ResConv, 1.000)
# ###1. Mix(5x5_MBConv1, 0.121)
# ###2. Mix(5x5_MBConv1, 0.125)
# ###3. Mix(3x3_MBConv1, 0.118)
# ###4. Mix(3x3_MBConv1, 0.101)
# ###5. Mix(3x3_MBConv1, 0.119)
# ###6. Mix(3x3_MBConv1, 0.126)
# ###7. Mix(5x5_MBConv1, 0.130)
# ###8. Mix(3x3_MBConv1, 0.125)
# os.system('python  SIRST_main_all.py --target_hardware flops --gpu 1,0 --warmup_epochs 300  --n_epochs 1500 --retrain_epoch 1500
#           --mode all_search_train --model_name Super_all')


####### 11-32-28 train 65.2%  Retrain 57.5%   代码错了，搞了9次下采样
# ###0. Mix(3x3_ResConv, 1.000)
# ###1. Mix(3x3_ResConv, 1.000)
# ###2. Mix(3x3_ResConv, 1.000)
# ###3. Mix(3x3_ResConv, 1.000)
# ###4. Mix(3x3_ResConv, 1.000)
# ###5. Mix(3x3_ResConv, 1.000)
# ###6. Mix(3x3_ResConv, 1.000)
# ###7. Mix(3x3_ResConv, 1.000)
# ###8. Mix(3x3_ResConv, 1.000)
# os.system('python  SIRST_main_all.py --target_hardware gpu   --gpu 1,0 --warmup_epochs 300  --n_epochs 1500 --retrain_epoch 1500 '
#           '--mode all_search_train   --model_name Super_half '
#           '--candidates_type single ')
