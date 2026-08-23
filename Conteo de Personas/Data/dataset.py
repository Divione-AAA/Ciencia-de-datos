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
        self.grid_size = config.get("GRID_SIZE", 80)
        self.num_classes = config.get("NUM_CLASSES", 1)
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

    @staticmethod
    def encode_boxes_to_grid(boxes, grid_size, image_size, num_classes=1):
        """
        Convierte cajas YOLO (batch, N, 5) = [cls, cx, cy, w, h] normalizado
        en un tensor denso (batch, grid, grid, 5 + num_classes)
        con la forma [tx, ty, tw, th, obj, cls_onehot] que espera YOLOLoss.

        tx, ty = desplazamiento del centro dentro de la celda [0, 1)
        tw, th = ancho y alto normalizados [0, 1]
        """
        def encode_image(one):
            one = tf.cast(one, tf.float32)
            valid_idx = tf.squeeze(tf.where(one[:, 0] >= 0.0), axis=1)
            num_valid = tf.cast(tf.shape(valid_idx)[0], tf.int32)
            target = tf.zeros((grid_size, grid_size, 5 + num_classes), tf.float32)

            def cond(k, tgt):
                return k < num_valid

            def body(k, tgt):
                i = valid_idx[k]
                b = one[i]

                cx = b[1]
                cy = b[2]
                w = b[3]
                h = b[4]

                col = tf.minimum(
                    tf.cast(tf.math.floor(cx * grid_size), tf.int32),
                    grid_size - 1
                )
                row = tf.minimum(
                    tf.cast(tf.math.floor(cy * grid_size), tf.int32),
                    grid_size - 1
                )

                tx = cx * grid_size - tf.cast(col, tf.float32)
                ty = cy * grid_size - tf.cast(row, tf.float32)
                tw = w
                th = h

                cls = tf.one_hot(
                    tf.cast(b[0], tf.int32),
                    num_classes,
                    dtype=tf.float32
                )
                val = tf.concat([[tx, ty, tw, th, 1.0], cls], axis=0)

                tgt = tf.tensor_scatter_nd_update(tgt, [[row, col]], [val])
                return k + 1, tgt

            _, target = tf.while_loop(cond, body, (0, target))
            return target

        return tf.map_fn(encode_image, boxes, fn_output_signature=tf.float32)

    def prepare_for_training(self, dataset):
        "Transforma las cajas en targets densos para la red"
        def apply_fn(image, boxes):
            targets = self.encode_boxes_to_grid(
                boxes,
                self.grid_size,
                self.image_size,
                self.num_classes
            )
            return image, targets

        return dataset.map(apply_fn, num_parallel_calls=self.autotune)

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