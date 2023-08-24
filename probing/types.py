from enum import Enum

try:
    from typing import Literal  # type: ignore
except:
    from typing_extensions import Literal  # type: ignore


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
    FIRST = "first"
    LAST = "last"
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
