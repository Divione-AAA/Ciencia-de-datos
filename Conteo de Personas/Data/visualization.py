import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tensorflow as tf
import numpy as np


class Visualizer:
    "El objetivo es comprobar visualmente que las imagenes concuerden con las cajas de las etiquetas"
    def __init__(self, class_names=None):
        if class_names is None:
            class_names = ["person"]
        self.class_names = class_names

    def yolo_to_xyxy(self, box, image_width, image_height):
        "Conversion de coordenadas"
        cls, xc, yc, w, h = box
        xc *= image_width
        yc *= image_height
        w *= image_width
        h *= image_height
        xmin = xc - w / 2
        ymin = yc - h / 2
        xmax = xc + w / 2
        ymax = yc + h / 2

        return xmin, ymin, xmax, ymax, int(cls)

    def show(self, image, boxes):
        "Las muestra"
        if tf.is_tensor(image):
            image = image.numpy()

        h, w = image.shape[:2]
        fig, ax = plt.subplots(figsize=(8,8))
        ax.imshow(image)

        for box in boxes:

            # Ignorar padding
            if box[0] < 0:
                continue

            xmin, ymin, xmax, ymax, cls = self.yolo_to_xyxy(box,w,h)

            rect = patches.Rectangle(
                (xmin, ymin),
                xmax - xmin,
                ymax - ymin,
                linewidth=2,
                edgecolor="red",
                facecolor="none"
            )

            ax.add_patch(rect)

            ax.text(
                xmin,
                ymin-5,
                self.class_names[cls],
                color="white",
                fontsize=10,
                bbox=dict(facecolor="red")
            )

        plt.axis("off")
        plt.show()

    def show_batch(self,images,boxes,max_images=4):
        "Muestra los batches"
        n = min(max_images, images.shape[0])
        fig, axes = plt.subplots(1,n,figsize=(5*n,5))

        if n == 1:
            axes = [axes]

        for i in range(n):

            image = images[i].numpy()
            ax = axes[i]
            ax.imshow(image)
            h, w = image.shape[:2]

            for box in boxes[i]:

                if box[0] < 0:
                    continue

                xmin, ymin, xmax, ymax, cls = self.yolo_to_xyxy(box.numpy(),w,h)

                rect = patches.Rectangle(
                    (xmin, ymin),
                    xmax-xmin,
                    ymax-ymin,
                    linewidth=2,
                    edgecolor="lime",
                    facecolor="none"
                )

                ax.add_patch(rect)

                ax.text(
                    xmin,
                    ymin-3,
                    self.class_names[cls],
                    color="white",
                    fontsize=8,
                    bbox=dict(facecolor="green")
                )

            ax.axis("off")

        plt.tight_layout()

        plt.show()

    def show_predictions(self,image,boxes,scores):

        if tf.is_tensor(image):
            image = image.numpy()

        h, w = image.shape[:2]
        fig, ax = plt.subplots(figsize=(8,8))
        ax.imshow(image)

        for box, score in zip(boxes, scores):

            xmin, ymin, xmax, ymax, cls = self.yolo_to_xyxy(box,w,h)

            rect = patches.Rectangle(
                (xmin,ymin),
                xmax-xmin,
                ymax-ymin,
                linewidth=2,
                edgecolor="yellow",
                facecolor="none"
            )

            ax.add_patch(rect)
            texto = f"{self.class_names[cls]} {score:.2f}"

            ax.text(
                xmin,
                ymin-5,
                texto,
                color="black",
                bbox=dict(facecolor="yellow")
            )

        plt.axis("off")

        plt.show()