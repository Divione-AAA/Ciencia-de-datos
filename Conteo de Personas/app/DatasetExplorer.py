from collections import Counter
from pathlib import Path
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

    def analyze_split(self, split):

        """
        Analiza un conjunto específico.
        """

        image_dir = self.dataset_path / split / "images"
        label_dir = self.dataset_path / split / "labels"
        image_paths = sorted(image_dir.glob("*"))
        label_paths = sorted(label_dir.glob("*.txt"))

        return {
            "images": len(image_paths),
            "labels": len(label_paths),
            "image_paths": image_paths,
            "label_paths": label_paths
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
        Busca imágenes que no poseen archivo .txt.
        """

        print("\nSearching images without labels...\n")
        total = 0

        for split in self.splits:
            image_dir = self.dataset_path / split / "images"
            label_dir = self.dataset_path / split / "labels"
            images = image_dir.glob("*")
            for image in images:
                label = label_dir / (image.stem + ".txt")
                if not label.exists():
                    total += 1
                    print(label.name)

        print()
        print(f"Missing labels : {total}")

    def labels_without_images(self):

        """
        Busca etiquetas huérfanas.
        """

        print("\nSearching labels without image...\n")
        total = 0

        for split in self.splits:
            image_dir = self.dataset_path / split / "images"
            label_dir = self.dataset_path / split / "labels"
            labels = label_dir.glob("*.txt")
            for label in labels:
                found = False
                for extension in [
                    ".jpg",
                    ".jpeg",
                    ".png"
                ]:

                    image = image_dir / (label.stem + extension)
                    if image.exists():
                        found = True
                        break
                if not found:
                    total += 1
                    print(label.name)

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
            label_dir = self.dataset_path / split / "labels"
            for file in label_dir.glob("*.txt"):
                with open(file) as f:
                    lines = f.readlines()
                total_objects += len(lines)
                for line in lines:
                    cls = int(line.split()[0])
                    class_counter[cls] += 1

        print("\nOBJECT STATISTICS")
        print("="*60)
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
            image_dir = self.dataset_path / split / "images"
            for image_path in image_dir.glob("*"):
                image = cv2.imread(str(image_path))
                h, w = image.shape[:2]
                resolutions[(w, h)] += 1

        print("\nIMAGE RESOLUTIONS")
        for resolution, count in resolutions.items():
            print(resolution,"->",count)