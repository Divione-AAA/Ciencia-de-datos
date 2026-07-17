from tensorflow.keras import layers
import tensorflow as tf

class ResidualBlock(tf.keras.Model):
    "Bloque residual"
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
        x = self.bn1(x,training=training)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x,training=training)
        x = x + shortcut #Con esto la informacion original nunca se pierde
        x = self.relu(x)

        return x