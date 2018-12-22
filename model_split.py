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
            yield ([x1,x2,x3,x4,x5,x6,x7,x8],y1)

def plot_history(history):
    # Plot training & validation accuracy values
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(file_name + '_acc.png')

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(file_name + '_loss.png')

os.environ["CUDA_VISIBLE_DEVICES"] = '2'
file_name = os.path.basename(sys.argv[0]).split('.')[0]

# batch_size = 8
# epochs = 100

# nb_train_samples = 167
# nb_test_samples = 42

# train_datagen = ImageDataGenerator()
# test_datagen = ImageDataGenerator()

# dynamic_image_path = '/data3/dingqianggang/Big_Data/local/datasets/Dynamic_Image'

# cam1_train_path = os.path.join(dynamic_image_path, 'cam1', 'train')
# cam2_train_path = os.path.join(dynamic_image_path, 'cam2', 'train')
# cam3_train_path = os.path.join(dynamic_image_path, 'cam3', 'train')
# cam4_train_path = os.path.join(dynamic_image_path, 'cam4', 'train')
# cam5_train_path = os.path.join(dynamic_image_path, 'cam5', 'train')
# cam6_train_path = os.path.join(dynamic_image_path, 'cam6', 'train')
# cam7_train_path = os.path.join(dynamic_image_path, 'cam7', 'train')
# cam8_train_path = os.path.join(dynamic_image_path, 'cam8', 'train')

# cam1_test_path = os.path.join(dynamic_image_path, 'cam1', 'test')
# cam2_test_path = os.path.join(dynamic_image_path, 'cam2', 'test')
# cam3_test_path = os.path.join(dynamic_image_path, 'cam3', 'test')
# cam4_test_path = os.path.join(dynamic_image_path, 'cam4', 'test')
# cam5_test_path = os.path.join(dynamic_image_path, 'cam5', 'test')
# cam6_test_path = os.path.join(dynamic_image_path, 'cam6', 'test')
# cam7_test_path = os.path.join(dynamic_image_path, 'cam7', 'test')
# cam8_test_path = os.path.join(dynamic_image_path, 'cam8', 'test')

# cam1_train_generator = train_datagen.flow_from_directory(cam1_train_path, target_size=(224, 224), batch_size=batch_size)
# cam2_train_generator = train_datagen.flow_from_directory(cam2_train_path, target_size=(224, 224), batch_size=batch_size)
# cam3_train_generator = train_datagen.flow_from_directory(cam3_train_path, target_size=(224, 224), batch_size=batch_size)
# cam4_train_generator = train_datagen.flow_from_directory(cam4_train_path, target_size=(224, 224), batch_size=batch_size)
# cam5_train_generator = train_datagen.flow_from_directory(cam5_train_path, target_size=(224, 224), batch_size=batch_size)
# cam6_train_generator = train_datagen.flow_from_directory(cam6_train_path, target_size=(224, 224), batch_size=batch_size)
# cam7_train_generator = train_datagen.flow_from_directory(cam7_train_path, target_size=(224, 224), batch_size=batch_size)
# cam8_train_generator = train_datagen.flow_from_directory(cam8_train_path, target_size=(224, 224), batch_size=batch_size)

# cam1_test_generator = test_datagen.flow_from_directory(cam1_test_path, target_size=(224, 224), batch_size=batch_size)
# cam2_test_generator = test_datagen.flow_from_directory(cam2_test_path, target_size=(224, 224), batch_size=batch_size)
# cam3_test_generator = test_datagen.flow_from_directory(cam3_test_path, target_size=(224, 224), batch_size=batch_size)
# cam4_test_generator = test_datagen.flow_from_directory(cam4_test_path, target_size=(224, 224), batch_size=batch_size)
# cam5_test_generator = test_datagen.flow_from_directory(cam5_test_path, target_size=(224, 224), batch_size=batch_size)
# cam6_test_generator = test_datagen.flow_from_directory(cam6_test_path, target_size=(224, 224), batch_size=batch_size)
# cam7_test_generator = test_datagen.flow_from_directory(cam7_test_path, target_size=(224, 224), batch_size=batch_size)
# cam8_test_generator = test_datagen.flow_from_directory(cam8_test_path, target_size=(224, 224), batch_size=batch_size)

input = []
y_bottom = []

for i in range(8):
    input.append(Input(shape=(224, 224, 3)))
    y_bottom.append(ResNet18_bottom(i, input_tensor=input[i]))
    
y = ResNet18_top(x=y_bottom)
y = GlobalAveragePooling2D()(y)
output = Dense(2, activation='softmax')(y)

model = Model(input, output)
model.load_weights('./saved_model/train_154546363969_best_weights.h5')

model_slaves = []
for i in range(8):
    input_slave = Input(shape=(224, 224, 3));
    # y_mid = ResNet18_bottom(i, input_tensor=input[i])
    y_mid = ResNet18_bottom(i, input_tensor=input_slave)
    m = Model(input_slave, y_mid)
    m.layers[1].set_weights(model.layers[8+i].get_weights())
    m.layers[2].set_weights(model.layers[16+i].get_weights())
    m.layers[3].set_weights(model.layers[24+i].get_weights())
    m.layers[4].set_weights(model.layers[32+i].get_weights())
    m.layers[5].set_weights(model.layers[40+i].get_weights())
    m.layers[6].set_weights(model.layers[48+i].get_weights())
    m.layers[7].set_weights(model.layers[56+i].get_weights())
    model_slaves.append(m)

input_master = [Input(shape=(56, 56, 64)) for _ in range(8)]
y_master = ResNet18_top(x=input_master)
y_master = GlobalAveragePooling2D()(y_master)
output_master = Dense(2, activation='softmax')(y_master)
model_master = Model(input_master, output_master)

for i in range(81):
    model_master.layers[8+i].set_weights(model.layers[64+i].get_weights())

model_master.save_weights('./saved_model/model_master.h')
for i in range(8):
    model_slaves[i].save_weights('./saved_model/model_slave{}.h'.format(i))


def test1():
    result_all = model.predict_generator(multi_generator(cam1_test_generator, cam2_test_generator, cam3_test_generator, cam4_test_generator, cam5_test_generator, cam6_test_generator, cam7_test_generator, cam8_test_generator), steps=nb_train_samples // batch_size)
    for model_sla in model_slaves:
        result_slave = model_sla.predict_generator(cam1_test_generator)

    print("Initialization Done")

    multi_gen = multi_generator(cam1_test_generator, cam2_test_generator, cam3_test_generator, cam4_test_generator, cam5_test_generator, cam6_test_generator, cam7_test_generator, cam8_test_generator)
    for multi in multi_gen:
        break

    label_truth = multi[1]
    xs = multi[0]

    result_final = model.predict(xs)
    result_mids = []
    for i in range(8):
        r = model_slaves[i].predict(xs[i])
        result_mids.append(r)

    result_master = model_master.predict(result_mids)


def test2():
    result_all_cent = model.predict_generator(multi_generator(cam1_test_generator, cam2_test_generator, cam3_test_generator, cam4_test_generator, cam5_test_generator, cam6_test_generator, cam7_test_generator, cam8_test_generator), steps=nb_train_samples // batch_size)

    cam1_test_generator = test_datagen.flow_from_directory(cam1_test_path, target_size=(224, 224), batch_size=batch_size)
    cam2_test_generator = test_datagen.flow_from_directory(cam2_test_path, target_size=(224, 224), batch_size=batch_size)
    cam3_test_generator = test_datagen.flow_from_directory(cam3_test_path, target_size=(224, 224), batch_size=batch_size)
    cam4_test_generator = test_datagen.flow_from_directory(cam4_test_path, target_size=(224, 224), batch_size=batch_size)
    cam5_test_generator = test_datagen.flow_from_directory(cam5_test_path, target_size=(224, 224), batch_size=batch_size)
    cam6_test_generator = test_datagen.flow_from_directory(cam6_test_path, target_size=(224, 224), batch_size=batch_size)
    cam7_test_generator = test_datagen.flow_from_directory(cam7_test_path, target_size=(224, 224), batch_size=batch_size)
    cam8_test_generator = test_datagen.flow_from_directory(cam8_test_path, target_size=(224, 224), batch_size=batch_size)

    result1 = model_slaves[0].predict_generator(cam1_test_generator)
    result2 = model_slaves[1].predict_generator(cam2_test_generator)
    result3 = model_slaves[2].predict_generator(cam3_test_generator)
    result4 = model_slaves[3].predict_generator(cam4_test_generator)
    result5 = model_slaves[4].predict_generator(cam5_test_generator)
    result6 = model_slaves[5].predict_generator(cam6_test_generator)
    result7 = model_slaves[6].predict_generator(cam7_test_generator)
    result8 = model_slaves[7].predict_generator(cam8_test_generator)

    result_all_dist = model_master.predict([result1, result2, result3, result4, result5, result6, result7, result8])

    multi_gen = multi_generator(cam1_test_generator, cam2_test_generator, cam3_test_generator, cam4_test_generator, cam5_test_generator, cam6_test_generator, cam7_test_generator, cam8_test_generator)
    r1 = []
    r2 = []
    pos = 0
    for multi in multi_gen:
        xs = multi[0]
        result_final = model.predict(xs)
        result_mids = []
        for i in range(8):
            r = model_slaves[i].predict(xs[i])
            result_mids.append(r)
        result_master = model_master.predict(result_mids)
        r1.append(result_final)
        r2.append(result_master)
        print(pos)
        pos += 1
        if pos == 4:
            break

    from IPython import embed
    embed() 



