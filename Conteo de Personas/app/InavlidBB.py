import csv
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
        self.splits = ["train", "valid", "test"]

    def _rows(self, split):
        """
        Lee el _annotations.csv de una partición
        y lo convierte a formato YOLO normalizado.
        """
        csv_path = self.dataset_path / split / "_annotations.csv"
        if not csv_path.exists():
            return [], []
        rows = []
        files = []
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if not row or not row.get("filename"):
                    continue
                width = float(row["width"])
                height = float(row["height"])
                xmin = float(row["xmin"])
                ymin = float(row["ymin"])
                xmax = float(row["xmax"])
                ymax = float(row["ymax"])

                yolo = np.asarray([
                    0.0,
                    ((xmin + xmax) / 2.0) / width,
                    ((ymin + ymax) / 2.0) / height,
                    (xmax - xmin) / width,
                    (ymax - ymin) / height,
                ], dtype=np.float32)

                rows.append(yolo)
                files.append(csv_path)

        return rows, files

    def invalid_boxes(self):
        """
        Busca cajas cuyo ancho o alto
        sean menores o iguales que cero.
        """

        invalid = []

        for split in self.splits:

            rows, files = self._rows(split)

            for row, file in zip(rows, files):

                width = row[3]
                height = row[4]

                if width <= 0 or height <= 0:
                    invalid.append({"file": file,"width":width,"height":height})

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

            rows, files = self._rows(split)

            for row, file in zip(rows, files):

                x = row[1]
                y = row[2]
                w = row[3]
                h = row[4]

                if (x < 0 or x > 1 or y < 0 or y > 1 or w <= 0 or w > 1 or h <= 0 or h > 1):
                    invalid.append({"file": file,"values":row})

        return invalid

    def report_out_of_bounds(self):

        invalid = self.out_of_bounds_boxes()
        print("OUT OF BOUNDS BOXES")
        print(f"Total: {len(invalid)}")

        for box in invalid:
            print(box)