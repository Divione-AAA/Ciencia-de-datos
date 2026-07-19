"""
Implementa la pérdida encargada de enseñar al modelo
a reconocer la clase de cada objeto detectado.

En un detector YOLO moderno esta pérdida trabaja
únicamente sobre las celdas donde realmente existe
un objeto (objectness = 1).
"""

import tensorflow as tf

class ClassificationLoss(tf.keras.losses.Loss):
    """
    Calcula la pérdida de clasificación.
    """

    def __init__(self,from_logits=False,label_smoothing=0.0,focal_gamma=0.0,alpha=1.0,class_weights=None,name="classification_loss"):

        super().__init__(name=name)

        self.from_logits = from_logits
        self.label_smoothing = label_smoothing
        self.focal_gamma = focal_gamma
        self.alpha = alpha
        self.class_weights = class_weights
        self.bce = tf.keras.losses.BinaryCrossentropy(
            from_logits=from_logits,
            reduction=tf.keras.losses.Reduction.NONE,
            label_smoothing=label_smoothing
        )

    def call(self, y_true, y_pred):
        """
        Calcula la pérdida de clasificación.
        Parámetros
        ----------
        y_true
            Tensor (batch,n_classes)
            Ejemplo
                [[1,0,0],
                 [0,1,0]]

        y_pred
            Tensor (batch,n_classes)
            Ejemplo
                [[0.92,0.05,0.03],
                 [0.12,0.80,0.08]]
        """

        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        loss = self.bce(y_true, y_pred)

        if self.class_weights is not None:

            """
            Multiplicamos la pérdida de cada clase
            por su peso correspondiente.

            Esto es útil cuando algunas clases
            aparecen muy poco.
            """

            weights = tf.reduce_sum(
                y_true * self.class_weights,
                axis=-1
            )

            loss *= weights

        if self.focal_gamma > 0:

            """
            pt representa la probabilidad asignada
            a la clase correcta.

            Si la red acierta mucho

                pt ≈ 1

            Si falla

                pt ≈ 0
            """

            pt = tf.exp(-loss)

            focal_factor = self.alpha * tf.pow(
                1.0 - pt,
                self.focal_gamma
            )

            loss *= focal_factor

        return tf.reduce_mean(loss)