import os
import sys
import numpy as np
from time import time
from my_models import *

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Input, GlobalAveragePooling2D, Dense

os.environ["CUDA_VISIBLE_DEVICES"] = '3'

input_master = [Input(shape=(56, 56, 64)) for _ in range(8)]
y_master = ResNet18_top(x=input_master)
y_master = GlobalAveragePooling2D()(y_master)
output_master = Dense(2, activation='softmax')(y_master)
model_master = Model(input_master, output_master)
model_master.load_weights('./saved_model/model_master.h')

# input = []
# y_bottom = []

# for i in range(8):
#     input.append(Input(shape=(224, 224, 3)))
#     y_bottom.append(ResNet18_bottom(i, input_tensor=input[i]))

# y = ResNet18_top(x=y_bottom)
# y = GlobalAveragePooling2D()(y)
# output = Dense(2, activation='softmax')(y)

# model = Model(input, output)
# model.load_weights('./saved_model/train_weights.h5')

pos = 0
time0 = time()
print("Input is ready")
while(True):
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
	# print(result[0])

	if np.argmax(result[0]) == 1:
		print("{}s other".format(pos))
	else:
		print("{}s fall".format(pos))




