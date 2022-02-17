from enum import Enum
import torch


class Metric:
    def __init__(self, metric_name: Enum):
        self.metric_name = metric_name
        self.metric = self.get_metric(metric_name)

    def accuracy(self, predictions, true_labels):
        return torch.mean((predictions == true_labels).float())

    def get_metric(self, metric_name):
        if metric_name == "accuracy":
            return self.accuracy
