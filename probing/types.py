from enum import Enum
from typing import Literal


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


MetricName = Literal[MetricType.accuracy, MetricType.f1]
ClassifierName = Literal[
    ClassifierType.logreg, ClassifierType.mlp, ClassifierType.mdlClassifierType
]
ProbingName = Literal[ProbingType.layerwise]
AggregationName = Literal[AggregationType.cls, AggregationType.sum, AggregationType.avg]
