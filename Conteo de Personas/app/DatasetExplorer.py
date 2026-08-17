import csv
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2

class DatasetExplorer:
    """
    Esta clase realiza un análisis exploratorio del dataset utilizado para
    entrenar el detector de personas.
    """

    def __init__(self, dataset_path):

        """
        Constructor.
        Parameters
        ----------
        dataset_path : str
        """

        self.dataset_path = Path(dataset_path)
        self.splits = [
            "train",
            "valid",
            "test"
        ]

    def _labels(self, split):
        """
        Cuenta el número de anotaciones en el _annotations.csv
        de una partición.
        """
        csv_path = self.dataset_path / split / "_annotations.csv"
        if not csv_path.exists():
            return []
        with open(csv_path, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def analyze_split(self, split):

        """
        Analiza un conjunto específico.
        """

        image_dir = self.dataset_path / split
        image_paths = sorted(image_dir.glob("*.jpg"))
        labels = self._labels(split)

        return {
            "images": len(image_paths),
            "labels": len(labels),
            "image_paths": image_paths,
            "label_paths": labels
        }

    def dataset_summary(self):

        """
        Imprime un resumen completo del dataset.
        """

        print("DATASET SUMMARY")

        total_images = 0
        total_labels = 0

        for split in self.splits:
            info = self.analyze_split(split)
            total_images += info["images"]
            total_labels += info["labels"]
            print(f"\n{split.upper()}")
            print(f"Images : {info['images']}")
            print(f"Labels : {info['labels']}")

        print("TOTAL")
        print(f"Images : {total_images}")
        print(f"Labels : {total_labels}")

    def images_without_labels(self):

        """
        Busca imágenes que no poseen ninguna anotación.
        """

        print("\nSearching images without labels...\n")
        total = 0

        for split in self.splits:
            image_dir = self.dataset_path / split
            images = image_dir.glob("*.jpg")
            labeled = {row["filename"] for row in self._labels(split)}
            for image in images:
                if image.name not in labeled:
                    total += 1
                    print(image.name)

        print()
        print(f"Missing labels : {total}")

    def labels_without_images(self):

        """
        Busca etiquetas huérfanas.
        """

        print("\nSearching labels without image...\n")
        total = 0

        for split in self.splits:
            image_dir = self.dataset_path / split
            names = {p.name for p in image_dir.glob("*.jpg")}
            for row in self._labels(split):
                if row["filename"] not in names:
                    total += 1
                    print(row["filename"])

        print()
        print(f"Orphan labels : {total}")

    def object_statistics(self):

        """
        Cuenta cuántos objetos existen
        en todo el dataset.
        """

        total_objects = 0
        class_counter = Counter()

        for split in self.splits:
            for row in self._labels(split):
                total_objects += 1
                class_counter[row["class"]] += 1

        print("\nOBJECT STATISTICS")
        print(f"Total objects : {total_objects}")
        print()
        for cls, count in class_counter.items():
            print(f"Class {cls} : {count}")

    def image_resolution_statistics(self):

        """
        Calcula las resoluciones presentes
        en el dataset.
        """

        resolutions = Counter()

        for split in self.splits:
            image_dir = self.dataset_path / split
            for image_path in image_dir.glob("*.jpg"):
                image = cv2.imread(str(image_path))
                h, w = image.shape[:2]
                resolutions[(w, h)] += 1

        print("\nIMAGE RESOLUTIONS")
        for resolution, count in resolutions.items():
            print(resolution,"->",count)