"""
Representa una caja delimitadora (Bounding Box).
Una Bounding Box almacena toda la información de una detección.
"""
from dataclasses import dataclass


@dataclass
class BoundingBox:


    #Coordenadas
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    #Info de la prediccion
    confidence: float = 0.0
    class_id: int = 0
    class_name: str = "person"#En ingles porque si
    #Dimensiones
    @property
    def width(self):
        """
        Calcula el ancho de la caja.
        width = xmax - xmin
        """

        return self.xmax - self.xmin


    @property
    def height(self):
        """
        Calcula  la altura.
        """

        return self.ymax - self.ymin


    @property
    def area(self):
        """
        Área ocupada por la Bounding Box.
        """

        return self.width * self.height


    @property
    def center_x(self):
        """
        Coordenada X del centro.
        """

        return (self.xmin + self.xmax) / 2


    @property
    def center_y(self):
        """
        Coordenada Y del centro.
        """

        return (self.ymin + self.ymax) / 2


    def to_xywh(self):
        """
        Convierte xmin,ymin,xmax,ymax en x_center,y_center,width,height
        """

        return (self.center_x,self.center_y,self.width,self.height)


    def to_xyxy(self):
        """
        Devuelve el formato clásico.
        """

        return (self.xmin,self.ymin,self.xmax,self.ymax)


    def is_valid(self):
        """
        Comprueba si la caja tiene dimensiones válidas.
        """

        if self.width <= 0:
            return False

        if self.height <= 0:
            return False

        return True


    def copy(self):
        """
        Devuelve una copia de la Bounding Box.
        """

        return BoundingBox(xmin=self.xmin,ymin=self.ymin,xmax=self.xmax,ymax=self.ymax,confidence=self.confidence,class_id=self.class_id,class_name=self.class_name)

    #Esto es como un tostring en java
    def __str__(self):
        """
        Define cómo se imprime la Bounding Box.
        """

        return (f"BoundingBox("f"class={self.class_name}, "f"conf={self.confidence:.2f}, "f"xmin={self.xmin:.1f}, "f"ymin={self.ymin:.1f}, "f"xmax={self.xmax:.1f}, "f"ymax={self.ymax:.1f})")