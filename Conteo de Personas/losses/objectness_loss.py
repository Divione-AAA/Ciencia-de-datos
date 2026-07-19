"""
Implementa la pérdida encargada de enseñar al modelo
si existe o no un objeto en cada celda
"""

import tensorflow as tf

class ObjectnessLoss(tf.keras.losses.Loss):
    """
    Calcula la Binary Cross Entropy para objectness
    """
    def __init__(self,reduction=tf.keras.losses.Reduction.AUTO,name="objectness_loss"):

        super().__init__(reduction=reduction,name=name)
        self.loss = tf.keras.losses.BinaryCrossentropy(
            from_logits=False
        )

    def call(self,y_true,y_pred):
        """
        Calcula la pérdida de objectness
        Parámetros
        ----------
        y_true
            Tensor con etiquetas binarias
        y_pred
            Tensor con probabilidades
            predichas por el modelo
        """

        y_true = tf.cast(y_true,tf.float32)
        y_pred = tf.cast(y_pred,tf.float32)

        return self.loss(y_true,y_pred)