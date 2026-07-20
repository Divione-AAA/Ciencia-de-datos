import tensorflow as tf

from models.Backbone import Backbone
from models.Neck import Neck
from models.YoloHead import YOLOHead

class YOLO(tf.keras.Model):
    "Une todo lo crado anteriormente"
    def __init__(self,num_classes=1):
        super().__init__(name="YOLO")
        self.backbone = Backbone()
        self.neck = Neck()
        self.head = YOLOHead(num_classes=num_classes)

    def call(self,x,training=False):

        feature160, feature80, feature40 = self.backbone(x,training=training)
        fused_features = self.neck(feature80,feature40,training=training)
        predictions = self.head(fused_features,training=training)

        return predictions