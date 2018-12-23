import os
import sys
import numpy as np
import time
from my_models import *
import paramiko

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Input, GlobalAveragePooling2D, Dense

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

def inference_master(arr, pos):
    print("Input is ready")

    xs = []
    for i in range(8):
        if arr[i] == 1:
            x = np.load('./output_cache/featuremap_{0}_{1}.npy'.format(i, pos))
        else:
            x = np.zeros((1, 56, 56, 64))
        xs.append(x)
    result = model_master.predict(xs)
    # print(result[0])

    if np.argmax(result[0]) == 1:
        print("{}s other".format(pos))
    else:
        print("{}s fall".format(pos))

input_master = [Input(shape=(56, 56, 64)) for _ in range(8)]
y_master = ResNet18_top(x=input_master)
y_master = GlobalAveragePooling2D()(y_master)
output_master = Dense(2, activation='softmax')(y_master)
model_master = Model(input_master, output_master)
model_master.load_weights('./saved_model/model_master.h')

host_list = ['219.223.189.131', '219.223.190.150', '219.223.190.154', '219.223.190.155']

# clean
os.system("rm -f output_cache/*")
for i in range(4):
    i = 1
    os.system("ssh %s 'source ~/jump/jumprc ; rm -f output_cache/* ; rm -f ~/jump/inference_slave.py' "%(host_list[i]))
    os.system("scp inference_slave.py %s:~/jump" % (host_list[i]))
    os.system("ssh %s 'source ~/jump/jumprc ; cp ~/jump/inference_slave.py ./'" % (host_list[i]))

for i in range(4):
    os.system( "ssh %s 'source ~/jump/jumprc ; nohup python inference_slave.py %d' > /dev/null &" % (host_list[i], i*2) )
    os.system( "ssh %s 'source ~/jump/jumprc ; nohup python inference_slave.py %d' > /dev/null &" % (host_list[i], i*2+1) )

# lost test
# time.sleep(2)
# os.system("rm FeatureMap_1.txt")

flag = 1
test_time = 0
while flag:
    iter_n = 100
    MapList = [0 for i in range(8)]
    while iter_n:
        for num in range(8):
            if MapList[num]==0:
                tmp = os.system("test -e %s" % ('output_cache/featuremap_%d_%d.npy')%(num, test_time) )
                if tmp == 0:
                    MapList[num] = 1 # exist
        print(MapList)
        if sum(MapList) == 8:
            break
        iter_n -= 1
    if sum(MapList) == 0:
        print('no input in %ds'%(test_time) )
        break
    inference_master(MapList, test_time)
    time.sleep(0.8)
    test_time += 1


