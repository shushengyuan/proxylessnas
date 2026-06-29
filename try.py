


# meta      = 5
# iteration = 5
# total    = int((meta+1)*meta/2)
# total_id = [m for m in range(total)]
# a        = [[_ for _ in range(meta-m)] for m in range(meta)]

meta = 6
for i in range(int((meta+1)*meta/2)):
    if   i - meta  < 0:
        iteration = 0
        layer     = i
    elif i - (2*meta-1)  < 0:
        iteration = 1
        layer     = i - meta
    elif i - (3*meta-3) < 0:
        iteration = 2
        layer     = i - (2*meta-1)
    elif i - (4*meta-6) < 0:
        iteration = 3
        layer     = i - (3*meta-3)
    elif i - (5*meta-10) < 0:
        iteration = 4
        layer     = i - (4*meta-6)
    elif i - (6*meta-15) < 0:
        iteration = 5
        layer     = i - (5*meta-10)

    print(i, 'iteration:', iteration, 'layer:', layer)


# m = 0
# num = 0
# for i in range(len(total_id)):
#
#     if i == iteration-m:
#         m+=1
#         num = 0
#
#     else:
#         iteration = m
#         layer     =  i-
#         num +=1
#



# meta  = 5
# initial_iteration = 5
# total    = int((initial_iteration+1)*initial_iteration/2)
# total_id = [m for m in range(total)]
#
#
# m = 0
# for i in range(len(total_id)):
#     if i != initial_iteration:
#         iteration = meta-initial_iteration
#         layer     = m
#         m        += 1
#
#     initial_iteration-=1
#     print(i, 'iteration：', iteration, 'layer:', layer)




# initial_iteration = 5
# # blocks
# for i in range(15):
#     if i in [m for m in range(initial_iteration)]:
#         iteration  = 0
#         layer      = i
#         max_0      = layer+1
#     elif i in [max_0 + m for m in range(initial_iteration - 1)]:
#         iteration  = 1
#         layer      = i -max_0
#         max_1      = max_0+layer+1
#     elif i in [max_1 + m for m in range(initial_iteration - 2)]:
#         iteration  = 2
#         layer      = i -max_1
#         max_2      = max_1+layer+1
#     elif i in [max_2 + m for m in range(initial_iteration - 3)]:
#         iteration  = 3
#         layer      = i -max_2
#         max_3      = max_2+layer+1
#     elif i in [max_3 + m for m in range(initial_iteration - 4)]:
#         iteration  = 3
#         layer      = i -max_3
#         max_4      = max_3+layer+1
#     elif i in [max_4 + m for m in range(initial_iteration - 5)]:
#         iteration  = 3
#         layer      = i -max_4
#         max_5      = max_4+layer+1
#     elif i in [max_5 + m for m in range(initial_iteration - 6)]:
#         iteration  = 3
#         layer      = i -max_5
#         max_6      = max_5+layer+1
#
#     print(i, 'iteration:',iteration, 'layer:',layer)