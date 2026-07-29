import tensorflow as tf

class Dataset():

    def __init__(self):
        pass
    
    def dts(self, image_paths, label_paths):
        dataset = (
            tf.data.Dataset
            .from_tensor_slices((image_paths, label_paths))
            .shuffle(1000)
            .map(
                self.parse_sample,
                num_parallel_calls=tf.data.AUTOTUNE
            )
            .cache()
            .padded_batch(
                self.batch_size,
                padded_shapes=(
                    [self.image_size, self.image_size, 3],
                    [None, 5]
                ),
                padding_values=(
                    tf.constant(0, tf.float32),
                    tf.constant(-1, tf.float32)
                )
            )
            .prefetch(tf.data.AUTOTUNE)
        )

        return dataset