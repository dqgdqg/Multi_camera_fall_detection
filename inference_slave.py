import socket
import sys
import struct
import numpy as np

import os
import tensorflow as tf

from time import time
from my_models import *

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Input

os.environ["CUDA_VISIBLE_DEVICES"] = ''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

batch_size = 1

train_datagen = ImageDataGenerator()
test_datagen = ImageDataGenerator()

dynamic_image_path = 'datasets/Dynamic_Image'
slave_index = int(sys.argv[1])

cam_train_path = os.path.join(dynamic_image_path, 'cam{}'.format(slave_index+1), 'train')
cam_test_path = os.path.join(dynamic_image_path, 'cam{}'.format(slave_index+1), 'test')
cam_train_generator = train_datagen.flow_from_directory(cam_train_path, target_size=(224, 224), batch_size=batch_size, shuffle=False)
cam_test_generator = test_datagen.flow_from_directory(cam_test_path, target_size=(224, 224), batch_size=batch_size, shuffle=False)

input_slave = Input(shape=(224, 224, 3));
y_mid = ResNet18_bottom(slave_index, input_tensor=input_slave)
model_slave = Model(input_slave, y_mid)
model_slave.load_weights('./saved_model/model_slave{}.h'.format(slave_index))

time0 = time()

def inference_slave(x, pos):
    result_mid = model_slave.predict(x)
    file_name = 'featuremap_{0}_{1}'.format(slave_index, pos)
    return file_name, result_mid

def socket_client():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # Master ip and port *******************************
        s.connect(('219.223.190.251', 22348))
        # **************************************************
    except socket.error as msg:
        print(msg)
        sys.exit(1)
    print(s.recv(1024).decode('utf-8'))

    pos = 0
    for (x, y) in cam_test_generator:
        # *************************************************************
        filename, data = inference_slave(x, pos)
        # *************************************************************
        fhead = struct.pack("128sl",filename.encode('utf-8'),sys.getsizeof(data))
        s.send(fhead)
        s.sendall(data)
        pos += 1
        if pos == 41:
            break
        
    s.close()


if __name__ == '__main__':
    socket_client()
