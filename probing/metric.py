from enum import Enum
from typing import List, Callable
import torch


class Metric:
    def __init__(self, metric_name: Enum):
        self.metric_name = metric_name
        self.metric = self.get_metric(metric_name)

    def accuracy(self, predictions: List[int], true_labels: List[int]) -> float:
        return torch.mean((torch.tensor(predictions) == torch.tensor(true_labels)).float())

    def get_metric(self, metric_name: Enum) -> Callable:
        if metric_name == "accuracy":
            return self.accuracy
