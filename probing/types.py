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
ClassifierName = Literal[ClassifierType.logreg, ClassifierType.mlp, ClassifierType.mdl]
ProbingName = Literal[ProbingType.layerwise]
AggregationName = Literal[AggregationType.cls, AggregationType.sum, AggregationType.avg]
UDProbingTaskName = Literal[
    "conj_type",
    "gapping",
    "impersonal_sent",
    "ngram_shift",
    "obj_gender",
    "obj_number",
    "predicate_aspect",
    "predicate_voice",
    "sent_len",
    "subj_gender",
    "subj_number",
    "tree_depth",
    "word_content",
]
