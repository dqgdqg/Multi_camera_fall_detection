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
from keras.models import Model
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
            
def plot_history(history):
    # Plot training & validation accuracy values
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(os.path.join('images', file_name + '_acc.png'))

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(os.path.join('images', file_name + '_loss.png'))
    
tf.app.flags.DEFINE_string('optimizer', 'adam', 'adam / sgd / rmsprop')
tf.app.flags.DEFINE_string('gpu', '1', '0 / 1 / 2 / 3')
tf.app.flags.DEFINE_float('learning_rate', 0.0001, 'default: 0.0001')
tf.app.flags.DEFINE_string('log_dir', 'default', 'tensorboard log dir, default: default')
tf.app.flags.DEFINE_integer('batch_size', 8, 'batch_size, default: 8')
tf.app.flags.DEFINE_integer('epochs', 100, 'epochs, default: 100')
tf.app.flags.DEFINE_string('pooling', 'Average', 'pooling method, default: Average')

flags = tf.app.flags.FLAGS

os.environ["CUDA_VISIBLE_DEVICES"] = flags.gpu
file_name = os.path.basename(sys.argv[0]).split('.')[0] + '_' + str(int(time.time()*100))

batch_size = flags.batch_size
epochs = flags.epochs

nb_train_samples = 165
nb_test_samples = 41

train_datagen = ImageDataGenerator()
test_datagen = ImageDataGenerator()

dynamic_image_path = './datasets/Dynamic_Image'

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

cam1_train_generator = train_datagen.flow_from_directory(cam1_train_path, target_size=(224, 224), batch_size=batch_size, seed=123)
cam2_train_generator = train_datagen.flow_from_directory(cam2_train_path, target_size=(224, 224), batch_size=batch_size, seed=123)
cam3_train_generator = train_datagen.flow_from_directory(cam3_train_path, target_size=(224, 224), batch_size=batch_size, seed=123)
cam4_train_generator = train_datagen.flow_from_directory(cam4_train_path, target_size=(224, 224), batch_size=batch_size, seed=123)
cam5_train_generator = train_datagen.flow_from_directory(cam5_train_path, target_size=(224, 224), batch_size=batch_size, seed=123)
cam6_train_generator = train_datagen.flow_from_directory(cam6_train_path, target_size=(224, 224), batch_size=batch_size, seed=123)
cam7_train_generator = train_datagen.flow_from_directory(cam7_train_path, target_size=(224, 224), batch_size=batch_size, seed=123)
cam8_train_generator = train_datagen.flow_from_directory(cam8_train_path, target_size=(224, 224), batch_size=batch_size, seed=123)

cam1_test_generator = test_datagen.flow_from_directory(cam1_test_path, target_size=(224, 224), batch_size=batch_size, seed=123)
cam2_test_generator = test_datagen.flow_from_directory(cam2_test_path, target_size=(224, 224), batch_size=batch_size, seed=123)
cam3_test_generator = test_datagen.flow_from_directory(cam3_test_path, target_size=(224, 224), batch_size=batch_size, seed=123)
cam4_test_generator = test_datagen.flow_from_directory(cam4_test_path, target_size=(224, 224), batch_size=batch_size, seed=123)
cam5_test_generator = test_datagen.flow_from_directory(cam5_test_path, target_size=(224, 224), batch_size=batch_size, seed=123)
cam6_test_generator = test_datagen.flow_from_directory(cam6_test_path, target_size=(224, 224), batch_size=batch_size, seed=123)
cam7_test_generator = test_datagen.flow_from_directory(cam7_test_path, target_size=(224, 224), batch_size=batch_size, seed=123)
cam8_test_generator = test_datagen.flow_from_directory(cam8_test_path, target_size=(224, 224), batch_size=batch_size, seed=123)
'''
fw = open('gen_filenames.txt','w')

for i in cam1_test_generator:
    idx = (cam1_test_generator.batch_index - 1) * cam1_test_generator.batch_size
    fw.writelines(str(cam1_test_generator.filenames[idx : idx + cam1_test_generator.batch_size]) + '\n')
    fw.writelines(str(cam3_test_generator.filenames[idx : idx + cam1_test_generator.batch_size]) + '\n')
    fw.writelines(str(cam5_test_generator.filenames[idx : idx + cam1_test_generator.batch_size]) + '\n')
    fw.writelines(str(cam7_test_generator.filenames[idx : idx + cam1_test_generator.batch_size]) + '\n')
    fw.writelines('\n')
exit(0)
'''
input = []
y_bottom = []

for i in range(8):
    input.append(Input(shape=(224, 224, 3)))
    y_bottom.append(ResNet18_bottom(i, input_tensor=input[i]))

y = ResNet18_top(x=y_bottom, pooling=flags.pooling)
y = GlobalAveragePooling2D()(y)
output = Dense(2, activation='softmax')(y)

model = Model(input, output)
plot_model(model, to_file='model.png')

if flags.optimizer == 'adam':
    optimizer = Adam(lr=flags.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.001, amsgrad=False)
elif flags.optimizer == 'sgd':
    optimizer = SGD(lr=flags.learning_rate, momentum=0.9, decay=0.001, nesterov=True)
elif flags.optimizer == 'rmsprop':
    optimizer = RMSprop(lr=flags.learning_rate, rho=0.9, epsilon=None, decay=0.001)
    
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

checkpoint = ModelCheckpoint(filepath=os.path.join('saved_model', file_name+'_best_weights.h5'), monitor='val_acc', verbose=1, save_best_only=True)
tensorboard = TensorBoard(write_images=True, write_graph=True, log_dir=os.path.join('./logs', flags.log_dir))
callbacks = [checkpoint, tensorboard]

history_ft = model.fit_generator(
    multi_generator(cam1_train_generator, cam2_train_generator, cam3_train_generator, cam4_train_generator, cam5_train_generator, cam6_train_generator, cam7_train_generator, cam8_train_generator),
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=multi_generator(cam1_test_generator, cam2_test_generator, cam3_test_generator, cam4_test_generator, cam5_test_generator, cam6_test_generator, cam7_test_generator, cam8_test_generator),
    validation_steps=nb_test_samples // batch_size,
    callbacks=callbacks
)

model.save_weights(os.path.join('saved_model', file_name + '_weights.h5'))
plot_history(history_ft)
