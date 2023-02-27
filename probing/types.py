from enum import Enum
from typing import Literal

MetricName = Literal["accuracy", "f1"]
ClassifierName = Literal["logreg", "mlp", "mdl"]
ProbingName = Literal["layerwise"]
AggregationName = Literal["cls", "sum", "avg"]


class MetricType(str, Enum):
    accuracy: MetricName = "accuracy"
    f1: MetricName = "f1"


class ClassifierType(str, Enum):
    logreg: ClassifierName = "logreg"
    mlp: ClassifierName = "mlp"
    mdl: ClassifierName = "mdl"


class ProbingType(str, Enum):
    layerwise: ProbingName = "layerwise"


class AggregationType(str, Enum):
    cls: AggregationName = "cls"
    sum: AggregationName = "sum"
    avg: AggregationName = "avg"
