import os
import sys
import keras
import numpy as np
import cv2
import get_di
from time import time
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
            yield ([x1,x2,x3,x4,x5,x6,x7,x8],y1)

os.environ["CUDA_VISIBLE_DEVICES"] = '3'
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

input_master = [Input(shape=(56, 56, 64)) for _ in range(8)]
y_master = ResNet18_top(x=input_master)
y_master = GlobalAveragePooling2D()(y_master)
output_master = Dense(2, activation='softmax')(y_master)
model_master = Model(input_master, output_master)
model_master.load_weights('./saved_model/model_master.h')

input_ori = []
y_bottom = []

for i in range(8):
    input_ori.append(Input(shape=(224, 224, 3)))
    y_bottom.append(ResNet18_bottom(i, input_tensor=input_ori[i]))
    
y = ResNet18_top(x=y_bottom)
y = GlobalAveragePooling2D()(y)
output = Dense(2, activation='softmax')(y)

model = Model(input_ori, output)
model.load_weights('./saved_model/train_weights.h5')

def load_slaves(slave_index):
	input_slave = Input(shape=(224, 224, 3));
	y_mid = ResNet18_bottom(slave_index, input_tensor=input_slave)
	model_slave = Model(input_slave, y_mid)
	model_slave.load_weights('./saved_model/model_slave{}.h'.format(slave_index))
	return model_slave

model_slaves = [load_slaves(i) for i in range(8)]


def check3():
	test = multi_generator(cam1_test_generator, cam2_test_generator, cam3_test_generator, cam4_test_generator, cam5_test_generator, cam6_test_generator, cam7_test_generator, cam8_test_generator)
	pos = 0
	time0 = time()
	for (x, y) in test:
		result_mids = []
		for i in range(8):
			result_mid = model_slaves[i].predict(x[i])
			result_mids.append(result_mid)
			np.save('./output_cache/featuremap_{0}_{1}'.format(i, pos), result_mid)
		result_master = model_master.predict(result_mids)
		result_ori = model.predict(x)
		print(result_master[0], result_ori[0])
		pos += 1
		if pos == 41:
			break
	# check3 meets consistency



def check1():
	test = multi_generator(cam1_test_generator, cam2_test_generator, cam3_test_generator, cam4_test_generator, cam5_test_generator, cam6_test_generator, cam7_test_generator, cam8_test_generator)

	pos = 0
	time0 = time()
	print("Input is ready.")
	for (x_ori, y_ori) in test:
		arr = input()
		xs = []
		for i in range(8):
			if arr[i] == '1':
				x = np.load('./output_cache/featuremap_{0}_{1}.npy'.format(i, pos))
			else:
				x = np.zeros((1, 56, 56, 64))
			xs.append(x)
		result = model_master.predict(xs)
		pos += 1
		while(time() - time0 < 1.0):
			pass
		time0 = time()
		print(result[0])

		result_ori = model.predict(x_ori)
		print(result_ori[0])
		if pos > 40:
			break
	# but check1 doesnot meet consistency


def check2():
	test = multi_generator(cam1_test_generator, cam2_test_generator, cam3_test_generator, cam4_test_generator, cam5_test_generator, cam6_test_generator, cam7_test_generator, cam8_test_generator)
	pos = 0
	time0 = time()
	for (x, y) in test:
		result_mids = []
		for i in range(8):
			r = model_slaves[i].predict(x[i])
			result_mids.append(r)
		result_master = model_master.predict(result_mids)
		result_ori = model.predict(x)
		print(result_master[0], result_ori[0])
		pos += 1
		if pos > 40:
			break

	# check2 consistency meet.


# check1()

from IPython import embed
embed()








