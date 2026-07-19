"""
Implementa las pérdidas de regresión de Bounding Boxes
utilizadas por YOLO.
Este módulo utiliza las métricas geométricas definidas
en iou.py para construir funciones de pérdida
diferenciables.
"""

from losses.iou import IoU
import tensorflow as tf

class BBoxLoss(tf.keras.losses.Loss):
    """
    Calcula la pérdida de localización.
    Parámetros
    ----------
    loss_type
        "iou"
        "giou"
        "diou"
        "ciou"
    """

    def __init__(self,loss_type="ciou",reduction=tf.keras.losses.Reduction.AUTO,name="bbox_loss"):

        super().__init__(reduction=reduction,name=name)#constructor de la calse padre
        self.loss_type = loss_type.lower()

    def call(self,y_true,y_pred):
        """
        Calcula la pérdida entre
        Bounding Boxes reales y predichas.

        Parámetros
        ----------
        y_true
            Tensor (...,4)
        y_pred
            Tensor (...,4)
        """

        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        loss = tf.map_fn(
            self.compute_single_loss,
            (y_true, y_pred),
            fn_output_signature=tf.float32
        )

        # Promedio del batch
        return tf.reduce_mean(loss)

    def compute_single_loss(self, boxes):
        """
        Calcula la pérdida para una sola pareja
        de Bounding Boxes.
        """

        gt = boxes[0]
        pred = boxes[1]

        if self.loss_type == "iou":
            similarity = IoU.iou(gt, pred)

        elif self.loss_type == "giou":
            similarity = IoU.giou(gt, pred)

        elif self.loss_type == "diou":
            similarity = IoU.diou(gt, pred)

        elif self.loss_type == "ciou":
            similarity = IoU.ciou(gt, pred)

        else:
            raise ValueError(f"Tipo de pérdida desconocido: {self.loss_type}")

        return 1.0 - similarity