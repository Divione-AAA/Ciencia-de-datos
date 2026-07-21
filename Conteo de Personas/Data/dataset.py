from Data.transform import Transforms
from pathlib import Path
import tensorflow as tf

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

    def load_label(self, label_path):
        "Procesa las etiquetas de las imagenes"
        text = tf.io.read_file(label_path)#Lee las etiquetas y las guarda
        text = tf.strings.strip(text)
        lines = tf.strings.split(text, "\n")#Las divide
        values = tf.strings.split(lines)
        values = tf.strings.to_number(values,tf.float32)

        return values

    def parse_sample(self,image_path,label_path):
        "Toma una de ejemplo"
        image = self.load_image(image_path)
        boxes = self.load_label(label_path)
        image, boxes = self.transforms(image,boxes)

        return image, boxes

    def get_paths(self, dataset_path, split):
        "Obtiene rutas"
        dataset_path = Path(dataset_path)
        image_dir = dataset_path / split / "images"
        label_dir = dataset_path / split / "labels"
        image_paths = sorted(image_dir.glob("*"))
        label_paths = []

        for image in image_paths:
            label = label_dir / (image.stem + ".txt")
            label_paths.append(label)

        return image_paths, label_paths

    def create_dataset(self,dataset_path,split):
        "Crea el dataset de tensorflow"
        image_paths, label_paths = self.get_paths(dataset_path,split)

        #obtiene las ubicaciones
        image_paths = [str(x) for x in image_paths]
        label_paths = [str(x) for x in label_paths]

        dataset = tf.data.Dataset.from_tensor_slices((image_paths,label_paths))

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