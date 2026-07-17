import tensorflow as tf
from tensorflow.keras import layers


class Backbone(tf.keras.Model):

    def __init__(self):

        super().__init__()

        ###########################
        # Primera convolución
        ###########################

        self.stem = tf.keras.Sequential([

            layers.Conv2D(
                32,
                3,
                strides=2,
                padding="same",
                use_bias=False
            ),

            layers.BatchNormalization(),

            layers.ReLU()

        ])

        ###########################
        # Bloque 1
        ###########################

        self.down1 = layers.Conv2D(

            64,

            3,

            strides=2,

            padding="same",

            use_bias=False

        )

        self.bn1 = layers.BatchNormalization()

        self.res1 = ResidualBlock(64)

        ###########################
        # Bloque 2
        ###########################

        self.down2 = layers.Conv2D(

            128,

            3,

            strides=2,

            padding="same",

            use_bias=False

        )

        self.bn2 = layers.BatchNormalization()

        self.res2 = ResidualBlock(128)

        ###########################
        # Bloque 3
        ###########################

        self.down3 = layers.Conv2D(

            256,

            3,

            strides=2,

            padding="same",

            use_bias=False

        )

        self.bn3 = layers.BatchNormalization()

        self.res3 = ResidualBlock(256)

    def call(self, x, training=False):

        x = self.stem(x, training=training)

        x = self.down1(x)

        x = self.bn1(x, training=training)

        x = tf.nn.relu(x)

        x = self.res1(x, training=training)

        x = self.down2(x)

        x = self.bn2(x, training=training)

        x = tf.nn.relu(x)

        x = self.res2(x, training=training)

        x = self.down3(x)

        x = self.bn3(x, training=training)

        x = tf.nn.relu(x)

        x = self.res3(x, training=training)

        return x