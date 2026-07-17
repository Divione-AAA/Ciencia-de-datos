import tensorflow as tf
from tensorflow.keras import layers


class ResidualBlock(tf.keras.Model):

    def __init__(self, filters):

        super().__init__()

        self.conv1 = layers.Conv2D(
            filters,
            3,
            padding="same",
            use_bias=False
        )

        self.bn1 = layers.BatchNormalization()

        self.conv2 = layers.Conv2D(
            filters,
            3,
            padding="same",
            use_bias=False
        )

        self.bn2 = layers.BatchNormalization()

        self.relu = layers.ReLU()

    def call(self, x, training=False):

        shortcut = x

        x = self.conv1(x)

        x = self.bn1(x, training=training)

        x = self.relu(x)

        x = self.conv2(x)

        x = self.bn2(x, training=training)

        x = x + shortcut

        x = self.relu(x)

        return x