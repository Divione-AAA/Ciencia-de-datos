"""
Implementa las pérdidas de regresión de Bounding Boxes
utilizadas por YOLO.
Este módulo utiliza las métricas geométricas definidas
en iou.py para construir funciones de pérdida
diferenciables.

Las cajas se reciben en formato cxcywh: [cx, cy, w, h]
y se convierten internamente a xyxy para calcular el IoU.
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

        super().__init__(reduction=reduction,name=name)#constructor de la clase padre
        self.loss_type = loss_type.lower()

    @staticmethod
    def to_xyxy(boxes):
        """
        Convierte cajas (..., 4) de formato cxcywh a xyxy.
        """

        cx = boxes[..., 0]
        cy = boxes[..., 1]
        w = boxes[..., 2]
        h = boxes[..., 3]

        return tf.stack([
            cx - w / 2.0,
            cy - h / 2.0,
            cx + w / 2.0,
            cy + h / 2.0
        ], axis=-1)

    def similarity(self, gt, pred):
        """
        Calcula la similitud geométrica entre
        dos lotes de Bounding Boxes xyxy.
        """

        if self.loss_type == "iou":
            return IoU.iou(gt, pred)

        elif self.loss_type == "giou":
            return IoU.giou(gt, pred)

        elif self.loss_type == "diou":
            return IoU.diou(gt, pred)

        elif self.loss_type == "ciou":
            return IoU.ciou(gt, pred)

        else:
            raise ValueError(f"Tipo de pérdida desconocido: {self.loss_type}")

    def call(self,y_true,y_pred,weights=None):
        """
        Calcula la pérdida entre
        Bounding Boxes reales y predichas.

        Parámetros
        ----------
        y_true
            Tensor (..., 4) en formato cxcywh.
        y_pred
            Tensor (..., 4) en formato cxcywh.
        weights
            Máscara opcional con el mismo forma que (...) 
            que indica en qué posiciones se calcula la pérdida.
            En YOLO se usa la objectness para ignorar
            las celdas sin objeto.
        """

        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        #Aplana todo a pares (N, 4)
        shape = tf.shape(y_true)
        flat_true = tf.reshape(y_true, (-1, 4))
        flat_pred = tf.reshape(y_pred, (-1, 4))

        #cxcywh -> xyxy y similitud vectorizada
        loss = 1.0 - self.similarity(
            BBoxLoss.to_xyxy(flat_true),
            BBoxLoss.to_xyxy(flat_pred)
        )

        if weights is None:
            return tf.reduce_mean(loss)

        #Solo penaliza las celdas con objeto
        flat_weights = tf.cast(tf.reshape(weights, (-1,)), tf.float32)
        total = tf.reduce_sum(flat_weights)

        return tf.reduce_sum(loss * flat_weights) / (total + 1e-7)