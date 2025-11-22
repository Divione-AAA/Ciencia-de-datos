import tensorflow as tf
from tf.keras.metrics import binary_accuracy

class CustomAccuracy(tf.keras.losses.Metric):
    def __init__(self, name='custom_accuracy', factor=1):
        super(CustomAccuracy, self).__init__(name=name, dtype=tf.float32)
        self.factor = factor
        self.accuarcy = self.add_weight(name='accuarcy', initializer='zeros')
    def update_state(self, y_true, y_pred, sample_weight=None):
        self.accuarcy.assign(binary_accuracy(y_true, y_pred))
    def result(self):
        return self.accuarcy
    def reset(self):
        self.accuarcy.assign(0.)