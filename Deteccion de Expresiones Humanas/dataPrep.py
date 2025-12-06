from pydoc import classname
import tensorflow as tf

class DatasetPreparation():
    def __init__(self):
        pass
    def loadData(self, path, CONFIGURATION):
        data = tf.keras.preprocessing.image_dataset_from_directory(
            path,
            image_size=(CONFIGURATION["IM_SIZE"], CONFIGURATION["IM_SIZE"]),
            color_mode="rgb",
            classname=CONFIGURATION["CLASS_NAMES"],
            batchsize=CONFIGURATION["BATCH_SIZE"],
            batch_size=32,
            label_mode="int",
            labrls="inferred",
            shuffle=True,
            seed=CONFIGURATION["SEED"],
            followLinks=False,
            Interpolation="bilinear",
            validation_split=0.2,
            subset="both",
        )
        return data
    