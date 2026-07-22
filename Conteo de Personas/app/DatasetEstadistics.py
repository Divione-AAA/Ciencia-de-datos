from pathlib import Path
import matplotlib.pyplot as plt

class DatasetStatistics:
    """
    Esta clase contiene diferentes estadísticas del dataset
    Es importante porque si el conjunto de datos entre train y validation 
    tienen tamanos muy distintos habra mucho cambio entre epocas durante
    el entrenamiento
    """

    def __init__(self, dataset_path):
        """
        Parameters
        ----------
        dataset_path : str
            Ruta principal del dataset.
        """

        self.dataset_path = Path(dataset_path)

        # Particiones del dataset.
        self.splits = ["train","valid","test"]

    def count_images(self):
        """
        Cuenta el número de imágenes existentes
        en cada partición.
        """

        counts = {}

        #Recorremos cada partición.
        for split in self.splits:
            #Ruta donde están las imágenes.
            image_dir = self.dataset_path / split / "images"
            #Contamos todas las imágenes.
            total = len(list(image_dir.glob("*")))
            counts[split] = total

        return counts
    
    def print_statistics(self):
        """
        Imprime un resumen del número de imágenes.
        """

        counts = self.count_images()
        total_images = sum(counts.values())
        print("IMAGE DISTRIBUTION")

        for split, value in counts.items():
            percentage = (value / total_images) * 100
            print(
                f"{split.upper():10}"
                f"{value:6} imágenes"
                f" ({percentage:.2f}%)"
            )

        print("=" * 60)
        print(f"TOTAL: {total_images}")

    def plot_distribution(self):
        """
        Dibuja un gráfico de barras mostrando
        la distribución de imágenes.
        """

        counts = self.count_images()
        plt.figure(figsize=(8,5))
        plt.bar(counts.keys(),counts.values())
        plt.title("Número de imágenes por partición")
        plt.xlabel("Partición")
        plt.ylabel("Cantidad de imágenes")

        # Mostrar el valor encima de cada barra.
        for x, y in zip(counts.keys(),counts.values()):
            plt.text(x,y,str(y),ha="center",fontsize=11)

        plt.grid(axis="y")
        plt.show()