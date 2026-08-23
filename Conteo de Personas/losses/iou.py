"""
Funciones geométricas utilizadas por las pérdidas de YOLO
No calcula pérdidas directamente
Su objetivo es medir qué tan parecidas son dos Bounding Boxes
bbox_loss.py utilizará estas funciones
para construir la función de pérdida

Todas las operaciones están vectorizadas sobre el último eje,
por lo que funcionan con una sola caja (4,) o con lotes (..., 4).
El formato de las cajas es xyxy: [xmin, ymin, xmax, ymax].
"""

import tensorflow as tf

class IoU:

    """
    Biblioteca de operaciones geométricas entre Bounding Boxes.
    """

    @staticmethod
    def intersection(box1, box2):
        """
        Calcula el área de intersección entre dos cajas.

        Parámetros
        ----------
        box1 : Tensor (..., 4)
        box2 : Tensor (..., 4)

        Retorna
        -------
        Área de la región donde ambas cajas se superponen.
        """

        #El uso de tf.maximum y minimun hace que sigamos trabajando con tensores
        #en lugar de con objetos de python que ocurriria si usaramos min max,
        #lo que romperia el flujo de trabajo

        #Borde izquierdo de la intersección
        xmin = tf.maximum(box1[..., 0], box2[..., 0])

        #Borde superior
        ymin = tf.maximum(box1[..., 1], box2[..., 1])

        #Borde derecho
        xmax = tf.minimum(box1[..., 2], box2[..., 2])

        #Borde inferior
        ymax = tf.minimum(box1[..., 3], box2[..., 3])

        #Evitar áreas negativas
        width = tf.maximum(0.0, xmax - xmin)
        height = tf.maximum(0.0, ymax - ymin)

        return width * height

    ###############################################################

    @staticmethod
    def area(box):
        """
        Calcula el área de una o más Bounding Boxes.

        Parámetros
        ----------
        box : Tensor (..., 4)

        Retorna
        -------
        Área de la(s) caja(s).
        """

        width = tf.maximum(0.0, box[..., 2] - box[..., 0])

        height = tf.maximum(0.0, box[..., 3] - box[..., 1])

        return width * height

    ###############################################################

    @staticmethod
    def union(box1, box2):
        """
        Calcula el área de la unión.
        Unión = Área1 + Área2 - Intersección
        """

        area1 = IoU.area(box1)
        area2 = IoU.area(box2)
        inter = IoU.intersection(box1, box2)

        return area1 + area2 - inter

    ###############################################################

    @staticmethod
    def iou(box1, box2):
        """
        Calcula el Intersection over Union.
        IoU = Intersección / Unión
        El resultado siempre pertenece al intervalo [0,1]
        """

        inter = IoU.intersection(box1, box2)
        uni = IoU.union(box1, box2)

        return inter / (uni + 1e-7)

    ###############################################################

    @staticmethod
    def enclosing_box(box1, box2):
        """
        Calcula la caja mínima que contiene ambas Bounding Boxes.

        Esta función será utilizada posteriormente por GIoU,
        DIoU y CIoU.
        """

        xmin = tf.minimum(box1[..., 0], box2[..., 0])
        ymin = tf.minimum(box1[..., 1], box2[..., 1])
        xmax = tf.maximum(box1[..., 2], box2[..., 2])
        ymax = tf.maximum(box1[..., 3], box2[..., 3])

        return tf.stack([xmin, ymin, xmax, ymax], axis=-1)

    ###############################################################

    @staticmethod
    def giou(box1, box2):
        """
        Generalized IoU.
        Mejora IoU cuando las cajas no se tocan.
        """

        iou = IoU.iou(box1, box2)
        enclosing = IoU.enclosing_box(box1, box2)
        area_c = IoU.area(enclosing)
        union = IoU.union(box1, box2)

        return iou - ((area_c - union) / (area_c + 1e-7))#Usamos epsilon para evitar divisiones por cero que romperian la app

    ###############################################################

    @staticmethod
    def center(box):
        """
        Calcula el centro de una Bounding Box.
        Retorna (cx,cy)
        """

        cx = (box[..., 0] + box[..., 2]) / 2
        cy = (box[..., 1] + box[..., 3]) / 2

        return cx, cy

    ###############################################################

    @staticmethod
    def diou(box1, box2):
        """
        Distance IoU.
        Penaliza la distancia entre centros.
        """

        iou = IoU.iou(box1, box2)
        cx1, cy1 = IoU.center(box1)
        cx2, cy2 = IoU.center(box2)
        center_distance = (cx1 - cx2) ** 2 + (cy1 - cy2) ** 2
        enclosing = IoU.enclosing_box(box1, box2)
        w = enclosing[..., 2] - enclosing[..., 0]
        h = enclosing[..., 3] - enclosing[..., 1]
        diagonal = w ** 2 + h ** 2

        return iou - center_distance / (diagonal + 1e-7)

    ###############################################################

    @staticmethod
    def ciou(box1, box2):
        """
        Complete IoU.
        Además de IoU y distancia entre centros,
        también compara la relación ancho/alto.
        Es la variante utilizada por la mayoría
        de implementaciones modernas de YOLO.
        """

        diou = IoU.diou(box1, box2)

        w1 = box1[..., 2] - box1[..., 0]
        h1 = box1[..., 3] - box1[..., 1]
        w2 = box2[..., 2] - box2[..., 0]
        h2 = box2[..., 3] - box2[..., 1]

        v = (4 / (3.1415926535 ** 2)) * tf.square(
            tf.atan(w1 / (h1 + 1e-7)) - tf.atan(w2 / (h2 + 1e-7))
        )
        alpha = v / ((1 - IoU.iou(box1, box2)) + v + 1e-7)

        return diou - alpha * v