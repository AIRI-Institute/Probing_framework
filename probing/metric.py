from enum import Enum
from typing import List, Callable, Union, Dict
import torch
from sklearn.metrics import f1_score as f1


class Metric:
    def __init__(self, metric_names: Union[Enum, List[Enum]]):
        self.metric_names = metric_names
        self.metrics = self.get_metric(metric_names)

    def __call__(self, predictions: List[int], true_labels: List[int]) -> Dict[Enum, float]:
        res_metrics = {}
        for m, f in self.metrics.items():
            res_metrics[m] = f(predictions, true_labels).item()
        return res_metrics

    def accuracy(self, predictions: List[int], true_labels: List[int]) -> float:
        return torch.mean((torch.tensor(predictions) == torch.tensor(true_labels)).float())

    def f1_score(self, predictions: List[int], true_labels: List[int]) -> float:
        return f1(true_labels, predictions, average='weighted')

    def get_metric(self, metric_names: Union[List[Enum], Enum]) -> Dict[Enum, Callable]:
        metrics = {}
        if "accuracy" in metric_names:
            metrics["accuracy"] = self.accuracy
        if "f1" in metric_names:
            metrics["f1"] = self.f1_score
        
        if not metrics:
            raise NotImplementedError("None known metrics were provided")
        return metrics
