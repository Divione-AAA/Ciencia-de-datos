from tensorflow.keras.layers import Dense, InputLayer, Conv2D, MaxPooling2D, Dense
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.metrics import Accuracy, FalseNegatives, FalsePositives, TruePositives, TrueNegatives
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import Callback, CSVLogger, EarlyStopping, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras import layers, models
import tensorflow as tf


class ConvModel():

    """Modelo convulsional secuencial con multiples capas"""

    def compile(self,CONFIGURATION):
        """
        Docstring para compile:

        Compila el modelo y crea un sumario
        
        :param CONFIGURATION: Diccionario de configuraciones generales para el modelo
        """
        self.model = models.Sequential([
            layers.Input(shape=(CONFIGURATION["IM_SIZE"],CONFIGURATION["IM_SIZE"],3)),

            # Convertir RGB -> Grayscale usando fórmula perceptual
            layers.Lambda(
                lambda x: tf.image.rgb_to_grayscale(x),
                name="rgb_to_gray"
            ),
            # Normalización
            layers.Rescaling(1./255, name="rescale"),
            # Bloque 1
            layers.Conv2D(
                32, (3, 3),
                strides=(1, 1),
                padding='same',
                activation='relu',
                use_bias=True,
                kernel_regularizer=l2(1e-4),
                name="Conv1"
            ),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2), strides=(2, 2), name="Pool1"),
            # Bloque 2
            layers.Conv2D(
                64, (3, 3),
                strides=(1, 1),
                padding='same',
                activation='relu',
                use_bias=True,
                kernel_regularizer=l2(1e-4),
                name="Conv2"
            ),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2), strides=(2, 2), name="Pool2"),
            # Bloque 3
            layers.Conv2D(
                128, (3, 3),
                strides=(1, 1),
                padding='same',
                activation='relu',
                use_bias=True,
                kernel_regularizer=l2(1e-4),
                name="Conv3"
            ),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2), strides=(2, 2), name="Pool3"),
            # Bloque 4
            layers.Conv2D(
                256, (3, 3),
                strides=(1, 1),
                padding='same',
                activation='relu',
                use_bias=True,
                kernel_regularizer=l2(1e-4),
                name="Conv4"
            ),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2), strides=(2, 2), name="Pool4"),
            # Bloque 5
            layers.Conv2D(
                512, (3, 3),
                strides=(1, 1),
                padding='same',
                activation='relu',
                use_bias=True,
                kernel_regularizer=l2(1e-4),
                name="Conv5"
            ),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2), strides=(2, 2), name="Pool5"),
            # Cabeza de Clasificación
            layers.GlobalAveragePooling2D(name="GAP"),

            # apagado aleatorio de neuronas
            layers.Dropout(0.4, name="dropout"),
            layers.Dense(3, activation='softmax',
                         name="output_softmax")
        ])
        self.model.summary()
        self.model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
        )
    @classmethod
    def get_callbacks(self):
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,      
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ModelCheckpoint(
                "best_emotion_model.keras",
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.3,     
                patience=4,
                min_lr=1e-7,
                verbose=1
            ),
            #Medida de seguridad
            tf.keras.callbacks.TerminateOnNaN()
        ]
        return callbacks

    def train_model(self,train_ds, val_ds, epochs=50):
        history = self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=self.get_callbacks(),
            verbose=1
        )
        return history

