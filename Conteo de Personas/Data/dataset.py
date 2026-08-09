import csv
from pathlib import Path
import numpy as np
import tensorflow as tf
from Data.transform import Transforms

class PeopleDataset:

    def __init__(self, config):
        "Constructor, la configuracion inicial es la de config.py"
        self.image_size = config["IMAGE_SIZE"]
        self.batch_size = config["BATCH_SIZE"]
        self.shuffle = config.get("SHUFFLE", True)
        self.buffer_size = config.get("BUFFER_SIZE", 1000)
        self.autotune = tf.data.AUTOTUNE
        self.transforms = Transforms(image_size=self.image_size)

    def load_image(self, image_path):
        "Cargas las imagenes"

        image = tf.io.read_file(image_path)#Lee las imagenes
        image = tf.image.decode_jpeg(image,channels=3)#decode rgb
        image = tf.image.resize(image,(self.image_size,self.image_size))#Las redimensiona segun las configuraciones
        image = tf.cast(image, tf.float32)#Las castea
        image /= 255.0

        return image

    def parse_sample(self, image_path, boxes):
        "Toma una de ejemplo"
        image = self.load_image(image_path)
        image, boxes = self.transforms(image, boxes)

        return image, boxes

    def get_paths(self, dataset_path, split):

        image_dir = Path(dataset_path) / split
        image_paths = sorted(image_dir.glob("*.jpg"))

        if not image_paths:
            raise FileNotFoundError(
                f"No se encontraron imagenes .jpg en {image_dir}"
            )

        csv_path = image_dir / "_annotations.csv"
        if not csv_path.exists():
            raise FileNotFoundError(
                f"No se encontro el archivo de anotaciones {csv_path}"
            )

        boxes_by_name = {}
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if not row or not row.get("filename"):
                    continue
                width = float(row["width"])
                height = float(row["height"])
                xmin = float(row["xmin"])
                ymin = float(row["ymin"])
                xmax = float(row["xmax"])
                ymax = float(row["ymax"])

                box = np.asarray([
                    0.0,
                    ((xmin + xmax) / 2.0) / width,
                    ((ymin + ymax) / 2.0) / height,
                    (xmax - xmin) / width,
                    (ymax - ymin) / height,
                ], dtype=np.float32)

                boxes_by_name.setdefault(row["filename"], []).append(box)

        return image_paths, boxes_by_name

    def _generator(self, image_paths, boxes_by_name):
        "Generador de (ruta de imagen, cajas en formato YOLO)"
        for image_path in image_paths:
            filename = image_path.name
            boxes = boxes_by_name.get(filename, [])
            boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 5)
            yield str(image_path), boxes

    def create_dataset(self, dataset_path, split):
        "Crea el dataset de tensorflow"
        image_paths, boxes_by_name = self.get_paths(dataset_path, split)

        dataset = tf.data.Dataset.from_generator(
            lambda: self._generator(image_paths, boxes_by_name),
            output_signature=(
                tf.TensorSpec(shape=(), dtype=tf.string),
                tf.TensorSpec(shape=(None, 5), dtype=tf.float32),
            ),
        )

        if split == "train" and self.shuffle:
            dataset = dataset.shuffle(self.buffer_size)

        dataset = dataset.map(
            self.parse_sample,
            num_parallel_calls=self.autotune
        )

        dataset = dataset.padded_batch(
            self.batch_size,
            padded_shapes=([self.image_size,self.image_size,3],[None, 5]),
            padding_values=(tf.constant(0,tf.float32),tf.constant(-1,tf.float32))
        )

        dataset = dataset.prefetch(self.autotune)

        return dataset

    def load(self, dataset_path):
        "Carga el train, valid y test"
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