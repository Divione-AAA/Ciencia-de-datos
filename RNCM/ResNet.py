from tensorflow.keras.layers import Dense,InputLayer, Conv2D, MaxPooling2D, Dense
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras import layers, models
import tensorflow as tf

class CustomConv2D(layers):
    def __init__(self, n_filters, kernel_size, n_strides, padding = 'valid'):
        super(CustomConv2D, self).__init__(name = "custom_conv2d")
        self.conv = Conv2D(
            filter = n_filters,
            kernel_size = kernel_size,
            activation = 'relu',
            strides = n_strides,
            padding = padding,
        )
        self.batchNorm = BatchNormalization()

    def call(self, x, training = True):
        x = self.conv
        x = self.batchNorm(x, training)
        return x

class ResNet34(tf.keras.models):
    def __init__(self):
        super(ResNet34,self).__init__(name="ResNet34")
        self.conv_1 = CustomConv2D(64,7,2, padding="same")
        self.max_pool = MaxPooling2D(3,2)

        self.con_2_1 = ResidualBlock(64)
        self.con_2_2 = ResidualBlock(64)
        self.con_2_3 = ResidualBlock(64)

        self.con_3_1 = ResidualBlock(128,2)
        self.con_3_2 = ResidualBlock(128)
        self.con_3_3 = ResidualBlock(128)
        self.con_3_4 = ResidualBlock(128)