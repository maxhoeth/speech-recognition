import tensorflow as tf
from tensorflow.keras import layers, models, backend, initializers
import numpy as np


norm_layer = layers.Normalization()
norm_layer.adapt(np.zeros(shape=(16, 121, 120, 1)))
input_shape = (121, 120, 1)


def CNN_Bi_2convLSTM():

    model = models.Sequential([
        layers.Input(shape=input_shape, dtype='int32'),
        layers.Resizing(64, 64),
        # Normalize.

        norm_layer,
        layers.Conv2D(10, 9, strides=1, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Conv2D(10, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        # '''Bidirectional RNN'''
        layers.Bidirectional(layers.ConvLSTM1D(10, 5, return_sequences=True), name='1'),
        layers.Bidirectional(layers.ConvLSTM1D(10, 2, return_sequences=True), name='2'),
        layers.MaxPooling2D(),




        layers.Flatten(),
        layers.Dense(128, activation='relu'), layers.Dense(36, activation='softmax')
    ])
    return model

def CNN_Bi_3convLSTM():

    model = models.Sequential([
        layers.Input(shape=input_shape, dtype='int32'),
        layers.Resizing(64, 64),
        # Normalize.

        norm_layer,
        layers.Conv2D(10, 9, strides=1, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Conv2D(10, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        # '''Bidirectional RNN'''
        layers.Bidirectional(layers.ConvLSTM1D(10, 5, return_sequences=True), name='1'),
        layers.Bidirectional(layers.ConvLSTM1D(10, 2, return_sequences=True), name='2'),
        layers.Bidirectional(layers.ConvLSTM1D(10, 1, return_sequences=True), name='3'),
        layers.MaxPooling2D(),




        layers.Flatten(),
        layers.Dense(128, activation='relu'), layers.Dense(36, activation='softmax')
    ])
    return model


def CNN_Bi_1convLSTM_1LSTM():
    model = models.Sequential([
        layers.Input(shape=input_shape, dtype='int32'),
        layers.Resizing(64, 64),
        # Normalize.

        norm_layer,
        layers.Conv2D(10, (5, 1), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Conv2D(10, (2, 1), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        # '''Bidirectional RNN'''
        layers.Bidirectional(layers.ConvLSTM1D(10, 5, padding='same')),
        layers.BatchNormalization(),
        layers.Bidirectional(layers.LSTM(10)),

        layers.Flatten(),
        layers.Dense(128, activation='relu'), layers.Dense(36, activation='relu')
    ])

    return model


def CNN_2convLSTM():
    model = models.Sequential([
        layers.Input(shape=input_shape, dtype='int32'),
        layers.Resizing(64, 64),
        # Normalize.

        norm_layer,
        layers.Conv2D(10, (5, 1), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Conv2D(10, (2, 1), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        # '''RNN'''
        layers.ConvLSTM1D(10, 5, padding='same', return_sequences=True),
        layers.BatchNormalization(),
        layers.ConvLSTM1D(10, 2, padding='same', return_sequences=True),
        layers.BatchNormalization(),

        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(36, activation='relu')
    ])

    return model


def CNN():
    model = models.Sequential([
        layers.Input(shape=input_shape, dtype='int32'),
        layers.Resizing(64, 64),
        # Normalize.

        norm_layer,
        layers.Conv2D(32, 64, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 32, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Conv2D(20, 16, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Conv2D(10, 9, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Conv2D(10, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),



        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(36, activation='relu')
    ])

    return model

def residual_CNN_RNN():

    def conv2d_bn(X_input, filters, kernel_size, strides, padding='same', activation=None):

        X = layers.Conv2D(filters, kernel_size, strides, padding, kernel_initializer=initializers.glorot_uniform(seed=0))(X_input)
        X = layers.BatchNormalization()(X)

        if activation is not None:
            X = layers.Activation(activation)(X)

        return X

    def block_1(X_input):
        branch_1 = layers.AveragePooling2D(5, 1, 'same')(X_input)
        branch_1 = conv2d_bn(branch_1, 10, 1, 1, 'same', 'ReLU')

        branch_2 = layers.ConvLSTM1D(10, 5, padding='same', return_sequences=True)(X_input)
        branch_2 = layers.ConvLSTM1D(10, 2, padding='same', return_sequences=True)(X_input)

        X = tf.concat([branch_1, branch_2], axis=3)

        return X

    def block_2(X_input):
        branch_1 = layers.AveragePooling2D(3, 1, 'same')(X_input)

        branch_2 = layers.ConvLSTM1D(10, 5, padding='same', return_sequences=True)(X_input)
        branch_2 = layers.ConvLSTM1D(10, 2, padding='same', return_sequences=True)(X_input)

        X = tf.concat([branch_1, branch_2], axis=3)

        return X

    X_input = layers.Input(shape=input_shape, dtype='int32')
    X = layers.Resizing(64, 64)(X_input)

    X = block_1(X)
    X = block_2(X)

    X = layers.Flatten()(X)
    X = layers.Dense(128)(X)
    X = layers.Dense(36)(X)

    model = models.Model(inputs=X_input, outputs=X, name='Residual_CNN_RNN')

    return model

def CNN_Bi_2convLSTM_AT():

    x_input = layers.Input(shape=input_shape, dtype='int32')
    x = layers.Resizing(64, 64)(x_input)
    # Normalize.

    x = norm_layer(x)
    x = layers.Conv2D(10, (5, 1), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    #x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(10, (2, 1), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    #x = layers.MaxPooling2D()(x)


    # '''Bidirectional RNN'''
    x = layers.Bidirectional(layers.ConvLSTM1D(10, 5, return_sequences=True), name='1')(x)
    x = layers.Bidirectional(layers.ConvLSTM1D(10, 2), name='2')(x)

    #'''Attention'''
    squeeze = layers.Lambda(lambda x: x[:, 1])(x)
    query = layers.Dense(59)(squeeze)

    attScore = layers.Dot(axes=[1, 1])([query, x])
    attScore = layers.Softmax()(attScore)

    attVector = layers.Dot(axes=([1, 2]))([attScore, x])
    print(attVector.shape)
    x = layers.Dense(50, activation='relu')(attVector)
    x = layers.Dense(40)(x)

    outputs = layers.Dense(36, activation='softmax')(x)

    model = models.Model(inputs=x_input, outputs=outputs, name='Residual_CNN_RNN')

    return model
