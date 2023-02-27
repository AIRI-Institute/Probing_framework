from enum import Enum
from typing import Literal

MetricName = Literal["accuracy", "f1"]
ClassifierName = Literal["logreg", "mlp", "mdl"]
ProbingName = Literal["layerwise"]
AggregationName = Literal["cls", "sum", "avg"]


class MetricType(str, Enum):
    accuracy = "accuracy"
    f1 = "f1"


class ClassifierType(str, Enum):
    logreg = "logreg"
    mlp = "mlp"
    mdl = "mdl"


class ProbingType(str, Enum):
    layerwise = "layerwise"


class AggregationType(str, Enum):
    cls = "cls"
    sum = "sum"
    avg = "avg"
