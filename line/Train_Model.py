# -*- coding: utf-8 -*-

import os
import shutil
#import mobilenet_v1
import cnn_model
import paddle as paddle
import reader
import paddle.fluid as fluid
import numpy as np
from sys import argv
import getopt

#script, save_path = argv


path = os.path.split(os.path.realpath(__file__))[0]+"/.."
#script, vels = argv
opts,args = getopt.getopt(argv[1:],'-hH',['test_list=','train_list=','save_path='])
#print(opts)

test_list = "test.list"
train_list = "train.list"
save_path = "model_infer"


#camera = "/dev/video0"

for opt_name,opt_value in opts:
    if opt_name in ('-h','-H'):
        print("python3 Train_Model.py  --test_list=%s   --train_list=%s  --save_path=%s  "%(test_list , train_list , save_path))
        exit()

    if opt_name in ('--test_list'):
        test_list  = opt_value

    if opt_name in ('--train_list'):
        train_list = opt_value
        
    if opt_name in ('--save_path'):
        save_path = opt_value

   
test_list  = path + '/data/' + test_list
train_list  = path + '/data/' + train_list
save_path  = path + '/model/' + save_path


crop_size = 128
resize_size = 128


image = fluid.layers.data(name='image', shape=[3, crop_size, crop_size], dtype='float32')
label = fluid.layers.data(name='label', shape=[1], dtype='float32')

model = cnn_model.cnn_model(image)

cost = fluid.layers.square_error_cost(input=model, label=label)
avg_cost = fluid.layers.mean(cost)

# 获取训练和测试程序
test_program = fluid.default_main_program().clone(for_test=True)

# 定义优化方法
optimizer = fluid.optimizer.AdamOptimizer(learning_rate=0.01, regularization=fluid.regularizer.L2Decay(0.00005))

# optimizer = fluid.optimizer.RMSProp(
#         learning_rate=fluid.layers.piecewise_decay(boundaries, values),
#         regularization=fluid.regularizer.L2Decay(0.00005))



opts = optimizer.minimize(avg_cost)

# 获取自定义数据
train_reader = paddle.batch(reader=reader.train_reader(train_list, crop_size, resize_size), batch_size=32)
test_reader = paddle.batch(reader=reader.test_reader(test_list, crop_size), batch_size=32)

# 定义执行器
#place = fluid.CPUPlace()  # i
place = fluid.CUDAPlace(0)
exe = fluid.Executor(place)
# 进行参数初始化
exe.run(fluid.default_startup_program())

# 定义输入数据维度
feeder = fluid.DataFeeder(place=place, feed_list=[image, label])

# 训练
all_test_cost = []
for pass_id in range(50):
    # 进行训练
    for batch_id, data in enumerate(train_reader()):
        train_cost = exe.run(program=fluid.default_main_program(),
                            feed=feeder.feed(data),
                            fetch_list=[avg_cost])

        # 每100个batch打印一次信息
        if batch_id % 100 == 0:
            print('Pass:%d, Batch:%d, TrainCost:%0.5f' %
                  (pass_id, batch_id, train_cost[0]))

    # 进行测试
    test_costs = []

    for batch_id, data in enumerate(test_reader()):
        test_cost = exe.run(program=test_program,
                            feed=feeder.feed(data),
                            fetch_list=[avg_cost])
        print("::");
        print(test_cost);
        test_costs.append(test_cost[0])
    # 求测试结果的平均值
    test_cost = (sum(test_costs) / len(test_costs))
    all_test_cost.append(test_cost)


    #test_acc = (sum(test_accs) / len(test_accs))
    print('Test:%d, Cost:%0.5f' % (pass_id, test_cost))
    #save_path = 'infer_model/'
    # 保存预测模型

    if min(all_test_cost) >= test_cost:
        fluid.io.save_inference_model(save_path, feeded_var_names=[image.name], main_program=test_program, target_vars=[model], executor=exe,params_filename='params',model_filename='model')
        print('finally test_cost: {}'.format(test_cost))

