"""
dataset.py
==============================

Carga el dataset para un detector tipo YOLO.

Estructura esperada:

dataset/
│
├── train/
│   ├── images/
│   └── labels/
│
├── valid/
│   ├── images/
│   └── labels/
│
└── test/
    ├── images/
    └── labels/

Cada imagen debe tener un txt con el mismo nombre.
"""

from Data.parser import Transforms
from pathlib import Path
import tensorflow as tf


class PeopleDataset:

    def __init__(self, config):

        self.image_size = config["IMAGE_SIZE"]
        self.batch_size = config["BATCH_SIZE"]
        self.shuffle = config.get("SHUFFLE", True)
        self.buffer_size = config.get("BUFFER_SIZE", 1000)
        self.autotune = tf.data.AUTOTUNE

    ####################################################################
    # Leer imagen
    ####################################################################

    def load_image(self, image_path):

        image = tf.io.read_file(image_path)

        image = tf.image.decode_jpeg(
            image,
            channels=3
        )

        image = tf.image.resize(
            image,
            (self.image_size,
             self.image_size)
        )

        image = tf.cast(image, tf.float32)

        image /= 255.0

        return image

    ####################################################################
    # Leer etiquetas YOLO
    ####################################################################

    def load_label(self, label_path):

        text = tf.io.read_file(label_path)

        text = tf.strings.strip(text)

        lines = tf.strings.split(text, "\n")

        values = tf.strings.split(lines)

        values = tf.strings.to_number(
            values,
            tf.float32
        )

        return values

    ####################################################################
    # Leer una muestra completa
    ####################################################################

    def parse_sample(self,
                     image_path,
                     label_path):
        
        image = self.load_image(image_path)

        boxes = self.load_label(label_path)

        image, boxes = self.transforms(
            image,
            boxes
        )

        return image, boxes

    ####################################################################
    # Obtener rutas
    ####################################################################

    def get_paths(self, dataset_path, split):

        dataset_path = Path(dataset_path)

        image_dir = dataset_path / split / "images"

        label_dir = dataset_path / split / "labels"

        image_paths = sorted(image_dir.glob("*"))

        label_paths = []

        for image in image_paths:

            label = label_dir / (image.stem + ".txt")

            label_paths.append(label)

        return image_paths, label_paths

    ####################################################################
    # Crear Dataset TensorFlow
    ####################################################################

    def create_dataset(self,
                       dataset_path,
                       split):

        image_paths, label_paths = self.get_paths(
            dataset_path,
            split
        )

        image_paths = [str(x) for x in image_paths]
        label_paths = [str(x) for x in label_paths]

        dataset = tf.data.Dataset.from_tensor_slices(
            (
                image_paths,
                label_paths
            )
        )

        if split == "train" and self.shuffle:

            dataset = dataset.shuffle(
                self.buffer_size
            )

        dataset = dataset.map(
            self.parse_sample,
            num_parallel_calls=self.autotune
        )

        # IMPORTANTE:
        # Como cada imagen puede tener distinta cantidad de personas,
        # usamos padded_batch.

        dataset = dataset.padded_batch(

            self.batch_size,

            padded_shapes=(

                [self.image_size,
                 self.image_size,
                 3],

                [None, 5]

            ),

            padding_values=(

                tf.constant(
                    0,
                    tf.float32
                ),

                tf.constant(
                    -1,
                    tf.float32
                )

            )

        )

        dataset = dataset.prefetch(
            self.autotune
        )

        return dataset

    ####################################################################
    # Cargar train, valid y test
    ####################################################################

    def load(self, dataset_path):

        train = self.create_dataset(
            dataset_path,
            "train"
        )

        valid = self.create_dataset(
            dataset_path,
            "valid"
        )

        test = self.create_dataset(
            dataset_path,
            "test"
        )

        return train, valid, test