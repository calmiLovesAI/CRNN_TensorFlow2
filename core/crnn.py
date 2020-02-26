import tensorflow as tf
from configuration import char2index_map


class VGG(tf.keras.layers.Layer):
    def __init__(self):
        super(VGG, self).__init__()
        self.conv_1 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding="same")
        self.bn_1 = tf.keras.layers.BatchNormalization()
        self.maxpool_1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))

        self.conv_2 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding="same")
        self.bn_2 = tf.keras.layers.BatchNormalization()
        self.maxpool_2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))

        self.conv_3 = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding="same")
        self.bn_3 = tf.keras.layers.BatchNormalization()
        self.conv_4 = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding="same")
        self.bn_4 = tf.keras.layers.BatchNormalization()
        self.maxpool_3 = tf.keras.layers.MaxPool2D(pool_size=(2, 1), strides=(2, 1))

        self.conv_5 = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), strides=1, padding="same")
        self.bn_5 = tf.keras.layers.BatchNormalization()
        self.conv_6 = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), strides=1, padding="same")
        self.bn_6 = tf.keras.layers.BatchNormalization()
        self.maxpool_4 = tf.keras.layers.MaxPool2D(pool_size=(2, 1), strides=(2, 1))

        self.conv_7 = tf.keras.layers.Conv2D(filters=512, kernel_size=(2, 2), strides=(2, 1), padding="same")
        self.bn_7 = tf.keras.layers.BatchNormalization()

    def call(self, inputs, training=None, **kwargs):
        x = self.conv_1(inputs)
        x = self.bn_1(x, training=training)
        x = tf.nn.relu(x)
        x = self.maxpool_1(x)

        x = self.conv_2(x)
        x = self.bn_2(x, training=training)
        x = tf.nn.relu(x)
        x = self.maxpool_2(x)

        x = self.conv_3(x)
        x = self.bn_3(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv_4(x)
        x = self.bn_4(x, training=training)
        x = tf.nn.relu(x)
        x = self.maxpool_3(x)

        x = self.conv_5(x)
        x = self.bn_5(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv_6(x)
        x = self.bn_6(x, training=training)
        x = tf.nn.relu(x)
        x = self.maxpool_4(x)

        x = self.conv_7(x)
        x = self.bn_7(x, training=training)
        x = tf.nn.relu(x)

        return x


class LSTMLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(LSTMLayer, self).__init__()
        self.lstm_1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=256, return_sequences=True),
                                                    merge_mode="concat")
        self.bn_1 = tf.keras.layers.BatchNormalization()
        self.lstm_2 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=256, return_sequences=True),
                                                    merge_mode="concat")
        self.bn_2 = tf.keras.layers.BatchNormalization()

    def call(self, inputs, training=None, **kwargs):
        x = self.lstm_1(inputs, training=training)
        x = self.bn_1(x, training=training)
        x = self.lstm_2(x, training=training)
        x = self.bn_2(x, training=training)
        return x


class CRNN(tf.keras.Model):
    def __init__(self):
        super(CRNN, self).__init__()
        self.NUM_CLASSES = len(char2index_map) + 1
        self.cnn_layer = VGG()
        self.lstm_layer = LSTMLayer()
        self.fc = tf.keras.layers.Dense(units=self.NUM_CLASSES)

    @staticmethod
    def __map_to_sequence(x):
        return tf.squeeze(input=x, axis=1)

    def call(self, inputs, training=None, mask=None):
        x = self.cnn_layer(inputs, training=training)
        x = self.__map_to_sequence(x)
        x = self.lstm_layer(x, training=training)
        x = self.fc(x)
        x = tf.transpose(a=x, perm=[1, 0, 2])
        return x

