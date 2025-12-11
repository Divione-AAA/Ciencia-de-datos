import tensorflow as tf
import os

class DatasetPreparation:
    
    def __init__(self):
        pass

    def load_split(self, path, CONFIG, subset):
        return tf.keras.preprocessing.image_dataset_from_directory(
            path,
            image_size=(CONFIG["IM_SIZE"], CONFIG["IM_SIZE"]),
            color_mode="rgb",
            class_names=CONFIG["CLASS_NAMES"],
            batch_size=CONFIG["BATCH_SIZE"],
            label_mode="int",
            shuffle=True,
            seed=CONFIG["SEED"],
            interpolation="bilinear",
            validation_split=0.2,
            subset=subset,      # "training" o "validation"
        )

    def load_all(self, path, CONFIG):
        # TRAIN + VALIDATION (divididos automáticamente)
        train = self.load_split(path, CONFIG, subset="training")
        val   = self.load_split(path, CONFIG, subset="validation")

        # TEST (no usa validation_split)
        test = tf.keras.preprocessing.image_dataset_from_directory(
            path.replace("train", "test"),
            image_size=(CONFIG["IM_SIZE"], CONFIG["IM_SIZE"]),
            color_mode="rgb",
            class_names=CONFIG["CLASS_NAMES"],
            batch_size=CONFIG["BATCH_SIZE"],
            shuffle=False,
        )

        return train, val, test
