from __future__ import print_function
import keras
from keras.models import Sequential,Model
from keras.layers import Dense, Dropout, Flatten, GlobalAveragePooling2D
from keras.layers import Conv2D, MaxPooling2D
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras import backend as K
import os
import random
import numpy as np
import cv2
batch_size = 1000
num_classes = 3
epochs = 50

# input image dimensions
img_rows, img_cols = 299, 299

# the data, split between train and test sets
#(x_train, y_train), (x_test, y_test) = mnist.load_data()
"""
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
"""
train_dir = 'train/'
test_dir = 'test/'
train_imgs = ['train/{}'.format(i) for i in os.listdir(train_dir)]
test_imgs = ['test/{}'.format(i) for i in os.listdir(test_dir)]
random.shuffle(train_imgs)
x_train = []
y_train = []
x_test = []
y_test = []
input_shape = (img_rows, img_cols, 1)
for image in train_imgs:
    x_train.append(cv2.resize(cv2.imread(image,cv2.IMREAD_COLOR),(img_rows,img_cols)))
    if 'COVID' in image:
        y_train.append(0)
    elif 'Norm' in image:
        y_train.append(1)
    else:
        y_train.append(2)
for image in test_imgs:
    x_test.append(cv2.resize(cv2.imread(image,cv2.IMREAD_COLOR),(img_rows,img_cols)))
    if 'COVID' in image:
        y_test.append(0)
    elif 'Norm' in image:
        y_test.append(1)
    else:
        y_test.append(2)
y_train = np.asarray(y_train)
y_test = np.asarray(y_test)
x_train = np.asarray(x_train)
x_test = np.asarray(x_test)
print (x_test)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 3)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 3)
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

base_model=InceptionResNetV2(weights='imagenet',include_top=False) #imports the mobilenet model and discards the last 1000 neuron layer.

x=base_model.output
x=GlobalAveragePooling2D()(x)
x=Dense(1024,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
x=Dense(1024,activation='relu')(x) #dense layer 2
x=Dense(512,activation='relu')(x) #dense layer 3
preds=Dense(3,activation='softmax')(x) #final layer with softmax activation
model = Model(inputs=base_model.inputs,outputs=preds)
for layer in model.layers[:20]:
    layer.trainable=False
for layer in model.layers[20:]:
    layer.trainable=True

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer='adam',
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
