import numpy as np
import tensorflow as tf


class CNN(tf.keras.Model):

    def __init__(self, height_size, weight_size, out_class, train=True):
        super(CNN, self).__init__()
        self.cnn_1 = tf.keras.layers.Conv2D(32, kernel_size=(3, 3), input_shape=(height_size, weight_size, 1),
                                            activation='relu')
        self.pool_1 = tf.keras.layers.MaxPool2D((2, 2))
        self.cnn_2 = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu')
        self.pool_2 = tf.keras.layers.MaxPool2D((2, 2))
        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(128)
        self.fc2 = tf.keras.layers.Dense(32)
        self.fc3 = tf.keras.layers.Dense(out_class)
        if train:
            self.dropout = tf.keras.layers.Dropout(0.5)
        else:
            self.dropout = tf.keras.layers.Dropout(0.0)
        self.relu = tf.keras.layers.ReLU()

    def call(self, inputs, training=None):
        # out = inputs
        out = tf.expand_dims(inputs, axis=-1)

        out = self.cnn_1(out)
        out = self.pool_1(out)
        out = self.cnn_2(out)
        out = self.pool_2(out)

        out = self.flatten(out)

        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.fc3(out)
        return out
