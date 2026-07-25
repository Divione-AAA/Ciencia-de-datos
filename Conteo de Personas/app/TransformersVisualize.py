import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import tensorflow as tf


class TransformVisualizer:
    """
    Permite visualizar el efecto de cada transformación
    aplicada sobre una imagen y sus bounding boxes.
    """

    def __init__(self, transforms):

        self.transforms = transforms
    
    def draw_boxes(self,image,boxes,title=""):
        """
        Dibuja una imagen junto con todas
        sus bounding boxes.
        """

        image = image.numpy()

        h, w = image.shape[:2]

        fig, ax = plt.subplots(figsize=(8,8))

        ax.imshow(image)

        for box in boxes:

            xmin = box[0] * w
            ymin = box[1] * h
            xmax = box[2] * w
            ymax = box[3] * h

            rect = patches.Rectangle(

                (xmin, ymin),

                xmax - xmin,

                ymax - ymin,

                linewidth=2,

                edgecolor="red",

                facecolor="none"

            )

            ax.add_patch(rect)

        ax.set_title(title)

        plt.show()
    
        def compare(self,image,boxes,transform,title=""):
            """
            Compara la imagen antes y después
            de aplicar una transformación.
            """

            transformed_image, transformed_boxes = transform(

                image,

                boxes

            )

            fig, axes = plt.subplots(

                1,

                2,

                figsize=(14,7)

            )

            for ax, img, bbs, name in zip(

                axes,

                [

                    image.numpy(),

                    transformed_image.numpy()

                ],

                [

                    boxes,

                    transformed_boxes

                ],

                [

                    "Original",

                    title

                ]

            ):

                h, w = img.shape[:2]

                ax.imshow(img)

                for box in bbs:

                    xmin = box[0] * w
                    ymin = box[1] * h
                    xmax = box[2] * w
                    ymax = box[3] * h

                    rect = patches.Rectangle(

                        (xmin, ymin),

                        xmax - xmin,

                        ymax - ymin,

                        edgecolor="red",

                        linewidth=2,

                        facecolor="none"

                    )

                    ax.add_patch(rect)

                ax.set_title(name)

            plt.show()