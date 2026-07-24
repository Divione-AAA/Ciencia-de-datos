from pathlib import Path
import numpy as np

class InvalidBoundingBoxes:

    def __init__(self, path):
        """
        Parameters
        ----------
        dataset_path : str
        """

        self.dataset_path = Path(path)

        
    def invalid_boxes(self):
        """
        Busca cajas cuyo ancho o alto
        sean menores o iguales que cero.
        """

        invalid = []

        for split in self.splits:

            label_dir = self.dataset_path / split / "labels"

            for label_file in label_dir.glob("*.txt"):

                data = np.loadtxt(label_file)

                if data.size == 0:
                    continue

                if data.ndim == 1:
                    data = np.expand_dims(data,0)

                for row in data:

                    width = row[3]
                    height = row[4]

                    if width <= 0 or height <= 0:
                        invalid.append({"file":label_file,"width":width,"height":height})

        return invalid
    
    def report_invalid_boxes(self):

        invalid = self.invalid_boxes()
        print("INVALID BOUNDING BOXES")
        print(f"Total invalid boxes: {len(invalid)}")

        for box in invalid:
            print(box)

    def out_of_bounds_boxes(self):
        """
        Busca anotaciones cuyos valores
        normalizados no pertenezcan
        al intervalo [0,1].
        """

        invalid = []

        for split in self.splits:

            label_dir = self.dataset_path / split / "labels"

            for label_file in label_dir.glob("*.txt"):

                data = np.loadtxt(label_file)

                if data.size == 0:
                    continue

                if data.ndim == 1:
                    data = np.expand_dims(data,0)

                for row in data:

                    x = row[1]
                    y = row[2]
                    w = row[3]
                    h = row[4]

                    if (x < 0 or x > 1 or y < 0 or y > 1 or w <= 0 or w > 1 or h <= 0 or h > 1):
                        invalid.append({"file":label_file,"values":row})

        return invalid
    
    def report_out_of_bounds(self):

        invalid = self.out_of_bounds_boxes()
        print("OUT OF BOUNDS BOXES")
        print(f"Total: {len(invalid)}")

        for box in invalid:
            print(box)