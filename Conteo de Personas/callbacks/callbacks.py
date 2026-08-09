"""
Callbacks utilizados durante el entrenamiento
del detector YOLO.
"""

import os
import tensorflow as tf

class Callbacks:
    """
    Fábrica de callbacks para entrenamiento.
    """

    def __init__(self, save_dir="checkpoints"):

        self.save_dir = save_dir#Direccion de guardado del mejor modelo
        os.makedirs(save_dir, exist_ok=True)

    def get_callbacks(self):

        callbacks = [
            #Detiene el entrenamiento si no mejora
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            #Guarda el mejor modelo
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(
                    self.save_dir,
                    "best_model.keras"
                ),
                monitor="val_loss",
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            ),
            #Permite definir estrategia
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.3,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            #Guarda historial
            tf.keras.callbacks.CSVLogger(
                os.path.join(
                    self.save_dir,
                    "training_log.csv"
                )
            ),
            #Tensorboard
            tf.keras.callbacks.TensorBoard(
                log_dir=os.path.join(
                    self.save_dir,
                    "logs"
                ),
                histogram_freq=1,
                write_graph=True,
                update_freq="epoch"
            ),
            tf.keras.callbacks.TerminateOnNaN()#Finaliza si hay un nan
        ]
        
        return callbacks