"""
Calcula Precision, Recall y F1-Score
La comparación entre cajas utiliza IoU
"""
from util.NMS import NonMaximumSuppression

class DetectionMetrics:

    def __init__(self, iou_threshold=0.50):
        self.iou_threshold = iou_threshold
        self.nms = NonMaximumSuppression(iou_threshold)

    def evaluate(self, predictions, ground_truth):

        matched_gt = set()
        tp = 0
        fp = 0

        for pred in predictions:

            best_iou = 0
            best_index = -1

            for i, gt in enumerate(ground_truth):

                if i in matched_gt:
                    continue

                iou = self.nms.compute_iou(pred, gt)

                if iou > best_iou:
                    best_iou = iou
                    best_index = i

            if best_iou >= self.iou_threshold:
                tp += 1
                matched_gt.add(best_index)
            else:
                fp += 1


        fn = len(ground_truth) - len(matched_gt)
        precision = self.precision(tp, fp)
        recall = self.recall(tp, fn)
        f1 = self.f1_score(precision, recall)

        return {"tp": tp,"fp": fp,"fn": fn,"precision": precision,"recall": recall,"f1_score": f1}

    def precision(self, tp, fp):

        """
        TP / (TP + FP)
        """

        if tp + fp == 0:
            return 0.0

        return tp / (tp + fp)

    def recall(self, tp, fn):

        """
        TP / (TP + FN)
        """

        if tp + fn == 0:
            return 0.0

        return tp / (tp + fn)

    def f1_score(self, precision, recall):

        """
        Media armónica.
        """

        if precision + recall == 0:
            return 0.0

        return (2 *precision *recall /(precision + recall))

    def print_metrics(self, metrics):

        """
        Imprime las métricas de forma legible.
        """

        print("=" * 40)
        print("RESULTADOS")
        print("=" * 40)
        print(f"TP         : {metrics['tp']}")
        print(f"FP         : {metrics['fp']}")
        print(f"FN         : {metrics['fn']}")
        print(f"Precision  : {metrics['precision']:.4f}")
        print(f"Recall     : {metrics['recall']:.4f}")
        print(f"F1 Score   : {metrics['f1_score']:.4f}")
        print("=" * 40)