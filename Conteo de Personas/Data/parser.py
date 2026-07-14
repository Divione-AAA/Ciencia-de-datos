"""
parser.py
------------------------------------
Lee imágenes y etiquetas en formato YOLO.

Responsabilidades:
    - Leer imágenes
    - Redimensionarlas
    - Normalizarlas
    - Leer archivos .txt de YOLO
    - Convertir etiquetas a tensores
"""

from pathlib import Path
import tensorflow as tf


class YOLOParser:

    def __init__(self,
                 image_size=640,
                 channels=3):

        self.image_size = image_size
        self.channels = channels

    ####################################################################
    # Leer imagen
    ####################################################################

    def read_image(self, image_path):

        image = tf.io.read_file(image_path)

        image = tf.image.decode_jpeg(
            image,
            channels=self.channels
        )

        return image

    ####################################################################
    # Resize
    ####################################################################

    def resize(self, image):

        image = tf.image.resize(
            image,
            (self.image_size,
             self.image_size)
        )

        return image

    ####################################################################
    # Normalización
    ####################################################################

    def normalize(self, image):

        image = tf.cast(image, tf.float32)

        image = image / 255.0

        return image

    ####################################################################
    # Leer etiquetas YOLO
    ####################################################################

    def read_label(self, label_path):

        text = tf.io.read_file(label_path)

        rows = tf.strings.strip(text)

        rows = tf.strings.split(rows, "\n")

        values = tf.strings.split(rows)

        values = tf.strings.to_number(
            values,
            tf.float32
        )

        return values

    ####################################################################
    # Función principal
    ####################################################################

    def parse(self,
              image_path,
              label_path):

        image = self.read_image(image_path)

        image = self.resize(image)

        image = self.normalize(image)

        labels = self.read_label(label_path)

        return image, labels