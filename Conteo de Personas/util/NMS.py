"""
Non-Maximum Suppression (NMS)
El objetivo es eliminar las cajas duplicadas que detectan el mismo objeto
"""

from util.BoundingBox import BoundingBox
from losses.iou import IoU

class NonMaximumSuppression:

    def __init__(self, iou_threshold=0.50):
        """
        Parámetros
        ----------
        iou_threshold

            Si dos cajas tienen un IoU superior a este valor,
            se considera que representan el mismo objeto.
        """

        self.iou_threshold = iou_threshold


    def compute_iou(self, box1, box2):
        """
        Calcula el Intersection over Union (IoU)
        entre dos BoundingBox.
        """

        return IoU.iou(box1, box2)


    def apply(self, boxes):
        """
        Ejecuta el algoritmo NMS.
        Entrada
            Lista de BoundingBox
        Salida
            Lista filtrada.
        """

        # Si no hay cajas
        if len(boxes) == 0:
            return []

        # Ordenar por confianza
        boxes = sorted(boxes,key=lambda b: b.confidence,reverse=True)
        selected_boxes = []

        while len(boxes) > 0:

            # Elegimos la mejor caja
            current = boxes.pop(0)
            selected_boxes.append(current)
            remaining_boxes = []

            for box in boxes:
                iou = self.compute_iou(current, box)
                #Mantener solo las cajas diferentes
                if iou < self.iou_threshold:
                    remaining_boxes.append(box)

            boxes = remaining_boxes

        return selected_boxes