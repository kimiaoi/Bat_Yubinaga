#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import numpy as np
from keras.layers.core import Dense, Dropout, Flatten, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.layers.normalization import BatchNormalization
import datetime
import keras
from keras.layers import Input
from keras.models import Model
from keras.backend import tensorflow_backend
import tensorflow as tf
import pickle
from keras import losses
from keras import metrics
config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
session = tf.Session(config=config)
tensorflow_backend.set_session(session)


def model2():
    inputs = Input(shape=(32, 8270, 18))

    x = Convolution2D(32, (5, 11), strides=(1, 2), padding='same')(inputs)
    x = BatchNormalization()(x, training=True)
    x = Activation('relu')(x)
    x = Convolution2D(32, (5, 11), strides=(1, 2), padding='same')(x) 
    x = BatchNormalization()(x, training=True) 
    x = Activation('relu')(x) 
    x = MaxPooling2D((2, 2), (2, 2), padding='same')(x)

    x = Convolution2D(64, (3, 9), strides=(1, 2), padding='same')(x) 
    x = BatchNormalization()(x, training=True) 
    x = Activation('relu')(x) 
    x = Convolution2D(64, (3, 9), strides=(1, 2), padding='same')(x)
    x = BatchNormalization()(x, training=True)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), (2, 2), padding='same')(x)

    x = Convolution2D(128, (3, 7), strides=(1, 2), padding='same')(x)
    x = BatchNormalization()(x, training=True)
    x = Activation('relu')(x)
    x = Convolution2D(128, (3, 7), strides=(1, 2), padding='same')(x)
    x = BatchNormalization()(x, training=True)
    x = Activation('relu')(x)
    x = Convolution2D(128, (3, 7), strides=(1, 2), padding='same')(x)
    x = BatchNormalization()(x, training=True)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), (2, 2), padding='same')(x)

    x = Convolution2D(256, (3, 5), strides=(1, 2), padding='same')(x)
    x = BatchNormalization()(x, training=True)
    x = Activation('relu')(x)
    x = Convolution2D(256, (3, 5), strides=(1, 2), padding='same')(x)
    x = BatchNormalization()(x, training=True)
    x = Activation('relu')(x)
    x = Convolution2D(256, (3, 5), strides=(1, 2), padding='same')(x)
    x = BatchNormalization()(x, training=True)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), (2, 2), padding='same')(x)


    x = Flatten()(x)

    x = Dense(5120)(x)
    x = Activation('relu')(x)

    x = Dense(5120)(x)
    x = Activation('relu')(x)

    x = Dropout(0.2)(x, training=True)

    x = Dense(1048)(x)
    x = Activation('relu')(x)

    x = Dropout(0.2)(x, training=True)

    outputs = Dense(3)(x)

    adam = Adam()

    model = Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer=adam,
                  loss=losses.mean_squared_error,
                    # metrics = ['mae','mape',metrics.categorical_crossentropy]
                  )

    model.summary()
    return model


def data_generator(datalist):
    data = np.load(datalist)
    batch_size = 10
    step = len(data) // batch_size

    while (True):
        # print("Epoch finish")
        np.random.shuffle(data)
        for i in range(step):
            X = []
            Y = []
            for j in range(batch_size):
                with open("pickledata3/" + data[i * batch_size + j], "rb") as f:
                    bat = pickle.load(f)
                X.append(bat['specgram'])
                Y.append(bat['bat1xyz'])

            # print("X : ", len(X))
            X = np.array(X)
            Y = np.array(Y)

            yield (X, Y)


traingen = data_generator("./newtrainlist.npy")
testgen = data_generator("./newtestlist.npy")

model = model2()
model.load_weights("./logs/newcsv_43.h5")
# model = load_model("../logs/model1_97.h5")

log_filepath = os.path.join(".", "tblogs", '-'.join(str(datetime.datetime.now()).split(' ')))
print(log_filepath)
tb_cb = keras.callbacks.TensorBoard(log_dir=log_filepath, write_graph=True)
cp = ModelCheckpoint('./logs/newcsv_{epoch:02d}.h5',
                     monitor='loss', save_best_only=True, mode='min')

model.fit_generator(traingen, 2396,
                    epochs=100,
                    callbacks=[tb_cb, cp],
                    )
# model.save("../20sampletest2.h5")