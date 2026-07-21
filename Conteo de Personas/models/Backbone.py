from tensorflow.keras import layers
from models import ResidualBlock
import tensorflow as tf

class Backbone(tf.keras.Model):
    "Extrae caracteristicas de la imagen, procesa bordes, texturas etc."
    def __init__(self):

        super().__init__()
        #Aprende los bordes
        self.stem = tf.keras.Sequential([
            layers.Conv2D(32,3,strides=2,padding="same",use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU()
        ])

        self.down1 = layers.Conv2D(64,3,strides=2,padding="same",use_bias=False)
        self.bn1 = layers.BatchNormalization()
        self.res1 = ResidualBlock(64)#Aprende texturas

        #Aprende esquinas
        self.down2 = layers.Conv2D(128,3,strides=2,padding="same",use_bias=False)
        self.bn2 = layers.BatchNormalization()
        self.res2 = ResidualBlock(128)
        
        # Aprende texturas     
        self.down3 = layers.Conv2D(256,3,strides=2,padding="same",use_bias=False)
        self.bn3 = layers.BatchNormalization()
        self.res3 = ResidualBlock(256)

    def call(self, x, training=False):

        # Stem
        x = self.stem(x, training=training)
        # Bloque 1
        x = self.down1(x)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.res1(x, training=training)
        feature1 = x
        # Bloque 2
        x = self.down2(x)
        x = self.bn2(x, training=training)
        x = tf.nn.relu(x)
        x = self.res2(x, training=training)
        feature2 = x
        # Bloque 3
        x = self.down3(x)
        x = self.bn3(x, training=training)
        x = tf.nn.relu(x)
        x = self.res3(x, training=training)
        feature3 = x

        return feature1, feature2, feature3
