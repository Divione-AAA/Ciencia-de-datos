import tensorflow as tf
from tensorflow.keras import layers

class YOLOv8:
    def YOLOv8(image_size=224,num_classes=3):
        ""
        inputs = tf.keras.Input(shape=(image_size,image_size,3))
        p3,p4,p5 = Backbone(inputs)
        p3,p4,p5 = FPN(p3,p4,p5)
        out3 = DetectionHead(p3,num_classes)
        out4 = DetectionHead(p4,num_classes)
        out5 = DetectionHead(p5,num_classes)
        return tf.keras.Model(inputs,[out3,out4,out5],name="YOLOv8")

    def ConvBNAct(x,filters,kernel=3,strides=1):
        ""
        x = layers.Conv2D(filters,kernel,strides=strides,padding="same",use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("swish")(x)
        return x

    def ResidualBlock(x,filters):
        ""
        shortcut = x
        x = ConvBNAct(x,filters,3)
        x = ConvBNAct(x,filters,3)
        x = layers.Add()([x,shortcut])
        return x

    def CSPBlock(x,filters,n=2):
        ""
        route1 = ConvBNAct(x,filters//2,1)
        route2 = ConvBNAct(x,filters//2,1)

        for _ in range(n):
            route2 = ResidualBlock(route2,filters//2)

        x = layers.Concatenate()([route1,route2])
        x = ConvBNAct(x,filters,1)

        return x

    def Backbone(inputs):
        ""
        x = ConvBNAct(inputs,32,3,2)
        x = ConvBNAct(x,64,3,2)
        p3 = CSPBlock(x,64)
        x = ConvBNAct(p3,128,3,2)
        p4 = CSPBlock(x,128)
        x = ConvBNAct(p4,256,3,2)
        p5 = CSPBlock(x,256)
        return p3,p4,p5

    def FPN(p3,p4,p5):
        ""
        p5_up = layers.UpSampling2D()(p5)
        p4 = layers.Concatenate()([p4,p5_up])
        p4 = ConvBNAct(p4,128)
        p4_up = layers.UpSampling2D()(p4)
        p3 = layers.Concatenate()([p3,p4_up])
        p3 = ConvBNAct(p3,64)

        return p3,p4,p5

    def DetectionHead(x,num_classes):
        ""
        bbox = layers.Conv2D(4,1,activation="sigmoid")(x)
        objectness = layers.Conv2D(1,1,activation="sigmoid")(x)
        classes = layers.Conv2D(num_classes,1,activation="softmax")(x)
        return layers.Concatenate()([bbox,objectness,classes])