from typing import Callable, Dict, List, Union

import torch
from sklearn.metrics import f1_score as f1
from sklearn.metrics import classification_report

from probing.types import MetricType


class Metric:
    def __init__(self, metric_names: Union[MetricType, List[MetricType]]):
        self.metric_names = metric_names

    def accuracy(self, predictions: List[int], true_labels: List[int]) -> float:
        return torch.mean(
            (torch.tensor(predictions) == torch.tensor(true_labels)).float()
        ).item()

    def f1_score(self, predictions: List[int], true_labels: List[int]) -> float:
        return f1(true_labels, predictions, average="weighted")
    
    def classification_report(self, predictions: List[int], true_labels: List[int]) -> float:
        return classification_report(true_labels, predictions, output_dict=True)

    def get_metrics_dict(self) -> Dict[MetricType, Callable]:
        metrics_dict = {}
        if MetricType("accuracy") in self.metric_names:
            metrics_dict[MetricType("accuracy")] = self.accuracy
        if MetricType("f1") in self.metric_names:
            metrics_dict[MetricType("f1")] = self.f1_score
        if MetricType("classification_report") in self.metric_names:
            metrics_dict[MetricType("classification_report")] = self.classification_report
        if not metrics_dict:
            raise NotImplementedError("None known metrics were provided")
        return metrics_dict

    def compute(
        self, predictions: List[int], true_labels: List[int]
    ) -> Dict[MetricType, float]:
        res_metrics = {}
        for m, f in self.get_metrics_dict().items():
            res_metrics[m] = f(predictions, true_labels)
        return res_metrics
