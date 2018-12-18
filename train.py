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
from keras.optimizers import SGD
from keras.utils import plot_model

# Path: 
raw_video = np.random.rand(8, 64, 224, 224, 3)
dynamic_image = np.zeros([8, 224, 224, 3])

for i in range(8):
    dynamic_image[i] = get_di.get_di(raw_video[i])
    # cv2.imwrite('test'+str(i)+'.png', di)

input = []
y_bottom = []

for i in range(8):
    input.append(Input(shape=(224, 224, 3)))
    y_bottom.append(ResNet18_bottom(i, input_tensor=input[i]))
    
y = ResNet18_top(x=y_bottom)

model = Model(input, y)

plot_model(model, to_file='model.png')


