import os
import sys
import keras
import numpy as np
import cv2
import get_di
import time
import matplotlib.pyplot as plt
from my_models import *
import tensorflow as tf

from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model, load_model
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, Input
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint, TensorBoard

# fw = open('gen_filenames.txt','w')

def multi_generator(generator1, generator2, generator3, generator4, generator5, generator6, generator7, generator8):
    while True:
        for (x1,y1),(x2,y2),(x3,y3),(x4,y4),(x5,y5),(x6,y6),(x7,y7),(x8,y8) in zip(generator1, generator2, generator3, generator4, generator5, generator6, generator7, generator8):
            # fw.writelines(str([y1,y2,y3,y4,y5,y6,y7,y8]) + '\n')
            # fw.writelines('\n')
            yield ([x1,x2,x3,x4,x5,x6,x7,x8],y1)
            
batch_size = 1

tf.app.flags.DEFINE_string('gpu', '1', '0 / 1 / 2 / 3')
tf.app.flags.DEFINE_string('model', 'None', 'Model ID.')
flags = tf.app.flags.FLAGS

os.environ["CUDA_VISIBLE_DEVICES"] = flags.gpu

train_datagen = ImageDataGenerator()
test_datagen = ImageDataGenerator()

dynamic_image_path = './datasets/Dynamic_Image'

cam1_test_path = os.path.join(dynamic_image_path, 'cam1', 'test')
cam2_test_path = os.path.join(dynamic_image_path, 'cam2', 'test')
cam3_test_path = os.path.join(dynamic_image_path, 'cam3', 'test')
cam4_test_path = os.path.join(dynamic_image_path, 'cam4', 'test')
cam5_test_path = os.path.join(dynamic_image_path, 'cam5', 'test')
cam6_test_path = os.path.join(dynamic_image_path, 'cam6', 'test')
cam7_test_path = os.path.join(dynamic_image_path, 'cam7', 'test')
cam8_test_path = os.path.join(dynamic_image_path, 'cam8', 'test')

cam1_test_generator = test_datagen.flow_from_directory(cam1_test_path, target_size=(224, 224), batch_size=batch_size, shuffle=False)
cam2_test_generator = test_datagen.flow_from_directory(cam2_test_path, target_size=(224, 224), batch_size=batch_size, shuffle=False)
cam3_test_generator = test_datagen.flow_from_directory(cam3_test_path, target_size=(224, 224), batch_size=batch_size, shuffle=False)
cam4_test_generator = test_datagen.flow_from_directory(cam4_test_path, target_size=(224, 224), batch_size=batch_size, shuffle=False)
cam5_test_generator = test_datagen.flow_from_directory(cam5_test_path, target_size=(224, 224), batch_size=batch_size, shuffle=False)
cam6_test_generator = test_datagen.flow_from_directory(cam6_test_path, target_size=(224, 224), batch_size=batch_size, shuffle=False)
cam7_test_generator = test_datagen.flow_from_directory(cam7_test_path, target_size=(224, 224), batch_size=batch_size, shuffle=False)
cam8_test_generator = test_datagen.flow_from_directory(cam8_test_path, target_size=(224, 224), batch_size=batch_size, shuffle=False)

model = load_model('./saved_model/train_%s_best_weights.h5'%(flags.model))

score = model.evaluate_generator(
    multi_generator(cam1_test_generator, cam2_test_generator, cam3_test_generator, cam4_test_generator, cam5_test_generator, cam6_test_generator, cam7_test_generator, cam8_test_generator),
    max_queue_size=41,
    verbose=1,
    steps=41
)

print(score)