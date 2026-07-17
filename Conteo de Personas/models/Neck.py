import tensorflow as tf
from tensorflow.keras import layers


class Neck(tf.keras.Model):
    "Mezcla caracteristicas, es neceseario porque combina informacion de distintas resoluciones"
    def __init__(self):

        super().__init__()

        #UpSampling2D: aumenta el tam
        self.up = layers.UpSampling2D(size=2,interpolation="nearest")
        #Concatena, no suma ni multiplica, pone una tras otras
        self.concat = layers.Concatenate()
        #Convulsiona para que aprenda como mezclar ambos modelos
        self.conv1 = layers.Conv2D(256,3,padding="same",use_bias=False)
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.ReLU()

    def call(self,low_feature,high_feature,training=False):

        high_feature = self.up(high_feature)
        x = self.concat([low_feature,high_feature])
        x = self.conv1(x)
        x = self.bn1(x,training=training)
        x = self.relu(x)

        return x