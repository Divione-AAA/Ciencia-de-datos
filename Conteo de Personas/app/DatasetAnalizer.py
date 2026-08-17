import csv
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

class DatasetAnalizer():

    def __init__(self, path, splits=None):
        self.dataset_path = Path(path)
        self.splits = splits or ["train", "valid", "test"]

    def _annotations(self, split):
        "Lee el _annotations.csv de una particion"
        csv_path = self.dataset_path / split / "_annotations.csv"
        rows = []
        if csv_path.exists():
            with open(csv_path, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    if row and row.get("filename"):
                        rows.append(row)
        return rows

    def people_per_image(self):

        people = Counter()
        for split in self.splits:
            for row in self._annotations(split):
                people[row["filename"]] += 1

        return list(people.values())

    def plot_people_distribution(self):

        """
        Dibuja un histograma del número
        de personas por imagen.
        """

        people = self.people_per_image()
        plt.figure(figsize=(9,5))
        plt.hist(people,bins=30,edgecolor="black")
        plt.title("Distribución de personas por imagen")
        plt.xlabel("Número de personas")
        plt.ylabel("Frecuencia")
        plt.grid(True)
        plt.show()

    def bbox_width_distribution(self):
        """
        Extrae el ancho normalizado de todas
        las bounding boxes.
        """

        widths = []

        for split in self.splits:
            for row in self._annotations(split):
                width = float(row["width"])
                w = (float(row["xmax"]) - float(row["xmin"])) / width
                widths.append(w)

        return widths

    def plot_width_distribution(self):
        widths = self.bbox_width_distribution()
        plt.figure(figsize=(9,5))
        plt.hist(widths,bins=40,edgecolor="black")
        plt.title("Distribución del ancho")
        plt.xlabel("Width")
        plt.ylabel("Frecuencia")
        plt.show()