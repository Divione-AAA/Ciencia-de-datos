from tensorflow.keras.layers import Conv2D, BatchNormalization
from tensorflow.keras import layers
from tensorflow.keras.layers import MaxPooling2D, GlobalAveragePooling2D, Dense
import tensorflow as tf

class CustomConv2D(layers.Layer):
    def __init__(self, n_filters, kernel_size, strides=1, padding="same"):
        super().__init__()
        self.conv = Conv2D(
            filters=n_filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            use_bias=False
        )
        self.bn = BatchNormalization()
        self.relu = layers.ReLU()

    def call(self, x, training=False):
        x = self.conv(x)
        x = self.bn(x, training=training)
        return self.relu(x)
    
class ResidualBlock(layers.Layer):
    def __init__(self, filters, strides=1):
        super().__init__()

        self.conv1 = CustomConv2D(filters, 3, strides)
        self.conv2 = Conv2D(
            filters,
            3,
            strides=1,
            padding="same",
            use_bias=False
        )
        self.bn2 = BatchNormalization()

        # Shortcut
        if strides != 1:
            self.shortcut = Conv2D(
                filters,
                1,
                strides=strides,
                padding="same",
                use_bias=False
            )
        else:
            self.shortcut = lambda x: x

        self.relu = layers.ReLU()

    def call(self, x, training=False):
        shortcut = self.shortcut(x)

        x = self.conv1(x, training=training)
        x = self.conv2(x)
        x = self.bn2(x, training=training)

        x = x + shortcut
        return self.relu(x)

class ResNet34(tf.keras.Model):
    def __init__(self, num_classes=1000):
        super().__init__()

        # Stem
        self.conv1 = CustomConv2D(64, 7, strides=2)
        self.pool = MaxPooling2D(pool_size=3, strides=2, padding="same")

        # Conv2_x
        self.block2 = [
            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64)
        ]

        # Conv3_x
        self.block3 = [
            ResidualBlock(128, strides=2),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128)
        ]

        # Conv4_x
        self.block4 = [
            ResidualBlock(256, strides=2),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256)
        ]

        # Conv5_x
        self.block5 = [
            ResidualBlock(512, strides=2),
            ResidualBlock(512),
            ResidualBlock(512)
        ]

        # Head
        self.gap = GlobalAveragePooling2D()
        self.fc = Dense(num_classes, activation="softmax")

    def call(self, x, training=False):
        x = self.conv1(x, training=training)
        x = self.pool(x)

        for block in self.block2:
            x = block(x, training=training)

        for block in self.block3:
            x = block(x, training=training)

        for block in self.block4:
            x = block(x, training=training)

        for block in self.block5:
            x = block(x, training=training)

        x = self.gap(x)
        return self.fc(x)
