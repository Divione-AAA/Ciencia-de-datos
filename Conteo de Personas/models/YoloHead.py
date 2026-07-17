import tensorflow as tf
from tensorflow.keras import layers


class YOLOHead(tf.keras.Model):

    def __init__(self, num_classes=1):

        super().__init__()
        self.num_classes = num_classes
        self.conv1 = layers.Conv2D(256,3,padding="same",use_bias=False)
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(256,3,padding="same",use_bias=False)
        self.bn2 = layers.BatchNormalization()
        self.relu = layers.ReLU()
        self.prediction = layers.Conv2D(filters=5 + num_classes,kernel_size=1,padding="same")

    def call(self,x,training=False):

        x = self.conv1(x)
        x = self.bn1(x,training=training)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x,training=training)
        x = self.relu(x)
        x = self.prediction(x)

        return x