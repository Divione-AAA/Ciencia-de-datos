"""
Decodifica las predicciones del modelo
"""
from util.BoundingBox import BoundingBox
import tensorflow as tf

class Decoder:

    def __init__(self,image_size=640, confidence_threshold=0.50):

        #Tamaño original de la imagen
        self.image_size = image_size
        #Minimo de confianza

        self.confidence_threshold = confidence_threshold

    def decode(self, prediction):

        """
        Convierte un tensor en BoundingBoxes.
        """

        boxes = []

        #obtiene dimensiones
        grid_h = prediction.shape[0]
        grid_w = prediction.shape[1]

        #Tamaño de cada celda
        stride_x = self.image_size / grid_w
        stride_y = self.image_size / grid_h

        #Recorrer toda la gril
        for row in range(grid_h):

            for col in range(grid_w):

                #Leer una predicción
                pred = prediction[row, col]

                tx = float(pred[0])
                ty = float(pred[1])
                tw = float(pred[2])
                th = float(pred[3])

                objectness = tf.sigmoid(pred[4]).numpy()
                class_score = tf.sigmoid(pred[5]).numpy()

                #Confianza final
                confidence = objectness * class_score

                #Ignorar cajas malas
                if confidence < self.confidence_threshold:
                    continue

                #Centro de la caja
                center_x = (col + tx) * stride_x
                center_y = (row + ty) * stride_y

                #Tamaño
                width = abs(tw)
                height = abs(th)

                #Conversión
                xmin = center_x - width / 2
                ymin = center_y - height / 2
                xmax = center_x + width / 2
                ymax = center_y + height / 2

                #Crear BoundingBox
                box = BoundingBox(xmin=xmin,ymin=ymin,xmax=xmax,ymax=ymax,confidence=float(confidence),class_id=0,class_name="person")

                if box.is_valid():
                    boxes.append(box)

        return boxes