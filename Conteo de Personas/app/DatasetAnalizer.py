import matplotlib.pyplot as plt
import numpy as np

class DatasetAnalizer():

    def __init__(self, path, splits):
        self.splits = splits
        self.dataset_path = path

    def people_per_image(self):

        people = []
        for split in self.splits:
            label_dir = self.dataset_path / split / "labels"
            for label in label_dir.glob("*.txt"):
                with open(label) as f:
                    lines = f.readlines()
                people.append(len(lines))

        return people
    
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
        Extrae el ancho de todas
        las bounding boxes.
        """

        widths = []

        for split in self.splits:
            label_dir = self.dataset_path / split / "labels"
            for file in label_dir.glob("*.txt"):
                data = np.loadtxt(file)
                if data.ndim == 1:
                    data = np.expand_dims(data,0)
                widths.extend(data[:,3])

        return widths
    
    def plot_width_distribution(self):
        widths = self.bbox_width_distribution()
        plt.figure(figsize=(9,5))
        plt.hist(widths,bins=40,edgecolor="black")
        plt.title("Distribución del ancho")
        plt.xlabel("Width")
        plt.ylabel("Frecuencia")
        plt.show()