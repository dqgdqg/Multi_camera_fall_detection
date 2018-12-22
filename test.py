import os
import sys
import keras
import numpy as np
import cv2
import get_di
from my_models import *

from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, Input
from keras.optimizers import SGD, Adam
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint

def multi_generator(generator1, generator2, generator3, generator4, generator5, generator6, generator7, generator8):
    while True:
        for (x1,y1),(x2,y2),(x3,y3),(x4,y4),(x5,y5),(x6,y6),(x7,y7),(x8,y8) in zip(generator1, generator2, generator3, generator4, generator5, generator6, generator7, generator8):
            yield ([x1,x2,x3,x4,x5,x6,x7,x8],[y1,y2,y3,y4,y5,y6,y7,y8])

os.environ["CUDA_VISIBLE_DEVICES"] = '2'
file_name = os.path.basename(sys.argv[0]).split('.')[0]

batch_size = 1
epochs = 100

nb_train_samples = 167
nb_test_samples = 42

train_datagen = ImageDataGenerator()
test_datagen = ImageDataGenerator()

dynamic_image_path = '/data3/dingqianggang/Big_Data/local/datasets/Dynamic_Image'

cam1_train_path = os.path.join(dynamic_image_path, 'cam1', 'train')
cam2_train_path = os.path.join(dynamic_image_path, 'cam2', 'train')
cam3_train_path = os.path.join(dynamic_image_path, 'cam3', 'train')
cam4_train_path = os.path.join(dynamic_image_path, 'cam4', 'train')
cam5_train_path = os.path.join(dynamic_image_path, 'cam5', 'train')
cam6_train_path = os.path.join(dynamic_image_path, 'cam6', 'train')
cam7_train_path = os.path.join(dynamic_image_path, 'cam7', 'train')
cam8_train_path = os.path.join(dynamic_image_path, 'cam8', 'train')

cam1_test_path = os.path.join(dynamic_image_path, 'cam1', 'test')
cam2_test_path = os.path.join(dynamic_image_path, 'cam2', 'test')
cam3_test_path = os.path.join(dynamic_image_path, 'cam3', 'test')
cam4_test_path = os.path.join(dynamic_image_path, 'cam4', 'test')
cam5_test_path = os.path.join(dynamic_image_path, 'cam5', 'test')
cam6_test_path = os.path.join(dynamic_image_path, 'cam6', 'test')
cam7_test_path = os.path.join(dynamic_image_path, 'cam7', 'test')
cam8_test_path = os.path.join(dynamic_image_path, 'cam8', 'test')

cam1_train_generator = train_datagen.flow_from_directory(cam1_train_path, target_size=(224, 224), batch_size=batch_size)
cam2_train_generator = train_datagen.flow_from_directory(cam2_train_path, target_size=(224, 224), batch_size=batch_size)
cam3_train_generator = train_datagen.flow_from_directory(cam3_train_path, target_size=(224, 224), batch_size=batch_size)
cam4_train_generator = train_datagen.flow_from_directory(cam4_train_path, target_size=(224, 224), batch_size=batch_size)
cam5_train_generator = train_datagen.flow_from_directory(cam5_train_path, target_size=(224, 224), batch_size=batch_size)
cam6_train_generator = train_datagen.flow_from_directory(cam6_train_path, target_size=(224, 224), batch_size=batch_size)
cam7_train_generator = train_datagen.flow_from_directory(cam7_train_path, target_size=(224, 224), batch_size=batch_size)
cam8_train_generator = train_datagen.flow_from_directory(cam8_train_path, target_size=(224, 224), batch_size=batch_size)

cam1_test_generator = test_datagen.flow_from_directory(cam1_test_path, target_size=(224, 224), batch_size=batch_size)
cam2_test_generator = test_datagen.flow_from_directory(cam2_test_path, target_size=(224, 224), batch_size=batch_size)
cam3_test_generator = test_datagen.flow_from_directory(cam3_test_path, target_size=(224, 224), batch_size=batch_size)
cam4_test_generator = test_datagen.flow_from_directory(cam4_test_path, target_size=(224, 224), batch_size=batch_size)
cam5_test_generator = test_datagen.flow_from_directory(cam5_test_path, target_size=(224, 224), batch_size=batch_size)
cam6_test_generator = test_datagen.flow_from_directory(cam6_test_path, target_size=(224, 224), batch_size=batch_size)
cam7_test_generator = test_datagen.flow_from_directory(cam7_test_path, target_size=(224, 224), batch_size=batch_size)
cam8_test_generator = test_datagen.flow_from_directory(cam8_test_path, target_size=(224, 224), batch_size=batch_size)

input = []
y_bottom = []

for i in range(8):
    input.append(Input(shape=(224, 224, 3)))
    y_bottom.append(ResNet18_bottom(i, input_tensor=input[i]))
    
y = ResNet18_top(x=y_bottom)
y = GlobalAveragePooling2D()(y)
output = Dense(2, activation='softmax')(y)

model = Model(input, output)
model.load_weights('./saved_model/train_weights.h5')

train = multi_generator(cam1_train_generator, cam2_train_generator, cam3_train_generator, cam4_train_generator, cam5_train_generator, cam6_train_generator, cam7_train_generator, cam8_train_generator)
pos = 0
ys = np.zeros((100, 8))
for (_, y) in train:
    for i in range(8):
        ys[pos, i] = y[i][0, 1]
    pos += 1
    if pos >= 100:
        break
        
from IPython import embed
embed() 



