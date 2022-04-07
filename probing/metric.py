from enum import Enum
from typing import List, Callable
import torch
from sklearn.metrics import f1_score as f1


class Metric:
    def __init__(self, metric_name: Enum):
        self.metric_name = metric_name
        self.metric = self.get_metric(metric_name)

    def __call__(self, predictions: List[int], true_labels: List[int]) -> Callable:
        return self.metric(true_labels, predictions)

    def accuracy(self, predictions: List[int], true_labels: List[int]) -> float:
        return torch.mean((torch.tensor(predictions) == torch.tensor(true_labels)).float())

    def f1_score(self, predictions: List[int], true_labels: List[int]) -> float:
        return f1(true_labels, predictions, average='weighted')

    def get_metric(self, metric_name: Enum) -> Callable:
        if metric_name == "accuracy":
            return self.accuracy
        elif metric_name == "f1":
            return self.f1_score
        else:
            raise NotImplementedError(f"Unknown metric: {metric_name}")
