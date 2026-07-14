import tensorflow as tf


class Transforms:

    def __init__(self,
                 image_size=640,
                 flip_probability=0.5):

        self.image_size = image_size
        self.flip_probability = flip_probability

    ######################################################
    # Resize
    ######################################################

    def resize(self, image, boxes):

        image = tf.image.resize(
            image,
            (self.image_size,
             self.image_size)
        )

        return image, boxes

    ######################################################
    # Normalize
    ######################################################

    def normalize(self, image, boxes):

        image = tf.cast(image, tf.float32)

        image /= 255.0

        return image, boxes

    ######################################################
    # Horizontal Flip
    ######################################################

    def horizontal_flip(self,
                        image,
                        boxes):

        p = tf.random.uniform(())

        if p < self.flip_probability:

            image = tf.image.flip_left_right(image)

            cls = boxes[:, 0]

            x = 1.0 - boxes[:, 1]

            y = boxes[:, 2]

            w = boxes[:, 3]

            h = boxes[:, 4]

            boxes = tf.stack(
                [
                    cls,
                    x,
                    y,
                    w,
                    h
                ],
                axis=1
            )

        return image, boxes

    ######################################################
    # Brightness
    ######################################################

    def brightness(self,
                   image,
                   boxes):

        image = tf.image.random_brightness(
            image,
            max_delta=0.15
        )

        image = tf.clip_by_value(
            image,
            0.0,
            1.0
        )

        return image, boxes

    ######################################################
    # Contrast
    ######################################################

    def contrast(self,
                 image,
                 boxes):

        image = tf.image.random_contrast(
            image,
            lower=0.8,
            upper=1.2
        )

        image = tf.clip_by_value(
            image,
            0.0,
            1.0
        )

        return image, boxes

    ######################################################
    # Pipeline completo
    ######################################################

    def __call__(self,
                 image,
                 boxes):

        image, boxes = self.resize(image, boxes)

        image, boxes = self.normalize(image, boxes)

        image, boxes = self.horizontal_flip(
            image,
            boxes
        )

        image, boxes = self.brightness(
            image,
            boxes
        )

        image, boxes = self.contrast(
            image,
            boxes
        )

        return image, boxes