from enum import Enum
#from typing import Literal
from typing_extensions import Literal


class MetricType(str, Enum):
    ACCURACY = "accuracy"
    F1 = "f1"
    CLASSIFICATION_REPORT = "classification_report"


class ClassifierType(str, Enum):
    LOGREG = "logreg"
    MLP = "mlp"
    MDL = "mdl"


class ProbingType(str, Enum):
    LAYERWISE = "layerwise"


class AggregationType(str, Enum):
    CLS = "cls"
    SUM = "sum"
    AVG = "avg"


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
