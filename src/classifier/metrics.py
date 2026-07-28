from __future__ import annotations

from dataclasses import dataclass

import torch
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
)


@dataclass
class ClassificationMetrics:
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    confusion_matrix: list[list[int]]
    class_report: dict


class MetricsCalculator:
    """
    Production classification metrics calculator.
    """


    @staticmethod
    def accuracy(
        predictions: torch.Tensor,
        labels: torch.Tensor,
    ) -> float:

        correct = (
            predictions == labels
        ).sum().item()

        return correct / len(labels)


    @classmethod
    def calculate(
        cls,
        predictions: torch.Tensor,
        labels: torch.Tensor,
    ) -> ClassificationMetrics:


        accuracy = cls.accuracy(
            predictions,
            labels,
        )


        y_true = labels.cpu().numpy()

        y_pred = predictions.cpu().numpy()


        precision, recall, f1, support = (
            precision_recall_fscore_support(
                y_true,
                y_pred,
                labels=[0, 1, 2],
                zero_division=0,
            )
        )


        cm = confusion_matrix(
            y_true,
            y_pred,
            labels=[0, 1, 2],
        )


        class_report = {

            "Normal": {
                "precision": float(precision[0]),
                "recall": float(recall[0]),
                "f1": float(f1[0]),
                "support": int(support[0]),
            },


            "Alert": {
                "precision": float(precision[1]),
                "recall": float(recall[1]),
                "f1": float(f1[1]),
                "support": int(support[1]),
            },


            "Critical": {
                "precision": float(precision[2]),
                "recall": float(recall[2]),
                "f1": float(f1[2]),
                "support": int(support[2]),
            },
        }


        return ClassificationMetrics(
            accuracy=accuracy,

            precision=float(
                precision.mean()
            ),

            recall=float(
                recall.mean()
            ),

            f1_score=float(
                f1.mean()
            ),

            confusion_matrix=cm.tolist(),

            class_report=class_report,
        )