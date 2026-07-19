"""
Implementa la función de pérdida completa utilizada
por el detector YOLO.
"""

from losses.bbox_loss import BBoxLoss
from losses.objectness_loss import ObjectnessLoss
from losses.classification_loss import ClassificationLoss
import tensorflow as tf

class YOLOLoss(tf.keras.losses.Loss):
    """
    Función de pérdida completa de YOLO.
    """

    def __init__(self,lambda_box=7.5,lambda_obj=1.0,lambda_cls=0.5,bbox_type="ciou"):

        super().__init__(name="YOLOLoss")

        self.lambda_box = lambda_box
        self.lambda_obj = lambda_obj
        self.lambda_cls = lambda_cls
        self.bbox_loss = BBoxLoss(
            loss_type=bbox_type
        )
        self.objectness_loss = ObjectnessLoss(
            focal_gamma=2.0,
            alpha=0.25,
            label_smoothing=0.0
        )
        self.classification_loss = ClassificationLoss(
            focal_gamma=2.0,
            alpha=0.25,
            label_smoothing=0.05
        )

    def call(self, y_true, y_pred):
        """
        Calcula la pérdida total.
        """

        #Separa cajas delimitadoras
        true_boxes = y_true[..., :4]
        pred_boxes = y_pred[..., :4]

        #Separa Objectness
        true_obj = y_true[..., 4:5]
        pred_obj = y_pred[..., 4:5]

        #Separa Clases
        true_cls = y_true[..., 5:]
        pred_cls = y_pred[..., 5:]

        #Calcula perdidas individuales
        bbox_loss = self.bbox_loss(
            true_boxes,
            pred_boxes
        )

        object_loss = self.objectness_loss(
            true_obj,
            pred_obj
        )

        classification_loss = self.classification_loss(
            true_cls,
            pred_cls
        )

        #Las combinma
        total_loss = (
            self.lambda_box * bbox_loss +
            self.lambda_obj * object_loss +
            self.lambda_cls * classification_loss
        )

        #Registra metricas en TensorBoard

        self.add_metric(
            bbox_loss,
            name="bbox_loss",
            aggregation="mean"
        )

        self.add_metric(
            object_loss,
            name="objectness_loss",
            aggregation="mean"
        )

        self.add_metric(
            classification_loss,
            name="classification_loss",
            aggregation="mean"
        )

        self.add_metric(
            total_loss,
            name="total_loss",
            aggregation="mean"
        )

        return total_loss