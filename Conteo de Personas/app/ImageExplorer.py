import csv
from pathlib import Path
import cv2
import matplotlib.pyplot as plt

class ImageExplorer():

    def __init__(self,path):
        self.dataset_path = Path(path)
        self.splits = ["train", "valid", "test"]

    def image_resolution_distribution(self):
        """
        Calcula la resolución de todas las imágenes del dataset.
        """

        widths = []
        heights = []

        for split in self.splits:

            image_dir = self.dataset_path / split

            for image_path in image_dir.glob("*.jpg"):
                image = cv2.imread(str(image_path))
                
                if image is None:
                    continue

                h, w = image.shape[:2]
                widths.append(w)
                heights.append(h)

        return widths, heights
    
    def plot_image_resolution_distribution(self):
        """
        Dibuja la distribución de resoluciones.
        """

        widths, heights = self.image_resolution_distribution()

        plt.figure(figsize=(14,5))
        plt.subplot(1,2,1)
        plt.hist(widths,bins=30,edgecolor="black")
        plt.title("Distribución del ancho")
        plt.xlabel("Pixeles")
        plt.ylabel("Frecuencia")
        plt.subplot(1,2,2)
        plt.hist(heights,bins=30,edgecolor="black")
        plt.title("Distribución del alto")
        plt.xlabel("Pixeles")
        plt.ylabel("Frecuencia")
        plt.tight_layout()
        plt.show()

    def images_without_annotations(self):
        """
        Busca imágenes que no posean ninguna anotación.
        """

        empty_annotations = []

        # Recorremos train, valid y test
        for split in self.splits:

            image_dir = self.dataset_path / split
            csv_path = image_dir / "_annotations.csv"
            labeled = set()
            if csv_path.exists():
                with open(csv_path, newline="", encoding="utf-8") as f:
                    for row in csv.DictReader(f):
                        if row and row.get("filename"):
                            labeled.add(row["filename"])

            for image_path in image_dir.glob("*.jpg"):

                # Si no tiene anotaciones, no hay objetos marcados
                if image_path.name not in labeled:
                    empty_annotations.append({"split": split,"image": image_path.name})

        return empty_annotations
    
    def report_images_without_annotations(self):
        """
        Imprime un reporte de imágenes
        sin anotaciones.
        """

        empty = self.images_without_annotations()

        print("IMÁGENES SIN ANOTACIONES")
        print(f"Total encontradas: {len(empty)}\n")

        for item in empty:
            print(f"[{item['split']}] "f"{item['image']}")

    def corrupted_images(self):
        """
        Intenta abrir todas las imágenes del dataset utilizando OpenCV.
        Si OpenCV devuelve None, la imagen probablemente está
        dañada o el formato no es válido.
        """

        corrupted = []

        for split in self.splits:

            image_dir = self.dataset_path / split

            for image_path in image_dir.glob("*.jpg"):
                image = cv2.imread(str(image_path))
                # OpenCV no pudo abrir la imagen
                if image is None:
                    corrupted.append({"split": split,"image": image_path.name})

        return corrupted
    
    def report_corrupted_images(self):
        """
        Imprime un reporte con todas
        las imágenes corruptas.
        """

        corrupted = self.corrupted_images()

        print("IMÁGENES CORRUPTAS")
        print(f"Total encontradas: {len(corrupted)}\n")

        for image in corrupted:
            print(f"[{image['split']}] "f"{image['image']}")