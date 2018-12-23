import os
import sys
import numpy as np
import tensorflow as tf

from time import time
from my_models import *

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Input

os.environ["CUDA_VISIBLE_DEVICES"] = ''

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

pos = 0
time0 = time()
for (x, y) in cam_test_generator:
    result_mid = model_slave.predict(x)
    '''
    while(time() - time0 < 1.0):
        pass
    '''
    time0 = time()
    file_name = './output_cache/featuremap_{0}_{1}'.format(slave_index, pos)
    np.save(file_name, result_mid)
    os.system( "scp %s 219.223.190.251:/data3/dingqianggang/Big_Data/local/output_cache" % (file_name+'.npy') )
    pos += 1
    if pos == 41:
        break

# from IPython import embed
# embed()


