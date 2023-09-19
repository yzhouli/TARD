import tensorflow as tf


class LSTM(tf.keras.Model):

    def __init__(self, h_dim, word_size, vector_size, out_class, train=True):
        super(LSTM, self).__init__()
        self.lstm_1 = tf.keras.layers.LSTM(h_dim, input_shape=(word_size, vector_size), return_sequences=True)
        self.lstm_2 = tf.keras.layers.LSTM(h_dim, return_sequences=False)
        self.fc1 = tf.keras.layers.Dense(128)
        self.fc2 = tf.keras.layers.Dense(32)
        self.fc3 = tf.keras.layers.Dense(out_class)
        if train:
            self.dropout = tf.keras.layers.Dropout(0.5)
        else:
            self.dropout = tf.keras.layers.Dropout(0.0)
        self.relu = tf.keras.layers.ReLU()

    def call(self, inputs, training=None):
        out = self.lstm_1(inputs)
        out = self.lstm_2(out)

        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.fc3(out)
        return out
