from typing import Dict, List, Optional, Union
import os
from enum import Enum
from fastapi import FastAPI
from pydantic import BaseModel

from probing.ud_parser.ud_parser import ConlluUDParser
from probing.pipeline import ProbingPipeline


app = FastAPI()
# parser = ConlluUDParser(shuffle = True, verbose = True)
# framework = ProbingPipeline(<params>)


class HFModel(str):
    mbert = 'bert-base-multilingual-cased'
    bloom1b3 = 'bigscience/bloom-1b3'
    mt5base = "google/mt5-base"
    mt5small = "google/mt5-small"
    gpt2 = "gpt2"
    xlmrbase = "xlm-roberta-base"


class Classifier(str):
    logreg='logreg'
    mlp='mlp'


class Metric(str, Enum):
    f1='f1'
    accuracy='accuracy'


class EmbeddingsAggregation(str):
    avg='avg'
    sum='sum'
    cls="cls"


class ProbingType (str):
    layer='layer'


class MLPparams(BaseModel):
    dropout: float = 0.2
    hidden_size: int = 256


class ProbingFrameworkStart(BaseModel):
    model_name: HFModel
    probing_type: ProbingType = "layer"
    classifier_name: Classifier = "logreg"
    metric_names: Union[Metric, List[Metric]] = ["f1", "accuracy"]
    embeddings_type: EmbeddingsAggregation = "cls"
    encoding_batch_size: int = 8
    classifier_batch_size: int = 32
    truncation: bool = False

    
class ProbingFrameworkRun(BaseModel):
    language: List[str]
    category: List[str]
    train_epochs: int = 10
    verbose: bool = True


class UDParserStart(BaseModel):
    shuffle: bool = True
    verbose: bool = True


class UDParserStartConvert(BaseModel):
    tr_path: Optional[os.PathLike] = None
    va_path: Optional[os.PathLike] = None
    te_path: Optional[os.PathLike] = None


# Loggings
class LogLevel(str, Enum):
    info = 'Info'
    error = 'Error'
    warning = "Warning"


class LogRecord(BaseModel):
    level: LogLevel
    message: str


# @app.post("/ud-parser-start", response_model=UDParserStartResponse)
# def ud_parser_start(body: UDParserStartRequest):
#     result = None#UDParserStartResponse()
#     return result

# @app.post("/ud-parser-status", response_model=UDParserStatusResponse)
# def ud_parser_status(body: UDParserStatusRequest):
#     result = None#UDParserStatusResponse()
#     return result

# @app.post("/probing-start", response_model=ProbingFrameworkStartResponse)
# def probing_start(body: ProbingFrameworkStartRequest):
#     result = None#ProbingFrameworkStartResponse()
#     return result

# @app.post("/probing-status", response_model=ProbingFrameworkStatusResponse)
# def probing_status(body: ProbingFrameworkStatusRequest):
#     result = None#ProbingFrameworkStartRequest()
#     return result
