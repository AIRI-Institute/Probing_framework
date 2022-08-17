from typing import Dict, Union, List

from enum import Enum
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class JobStatus(str, Enum):
    wip = 'wip'
    done = 'done'
    error = 'error'

class HFModel(str, Enum):
    bert = 'bert'
    bloom = 'bloom'

class LogLevel(str, Enum):
    info = 'Info'
    error = 'Error'
    warning = "Warning"

class LogRecord(BaseModel):
    level: LogLevel
    message: str

class ProbingType(str, Enum):
    logreg='logreg'
    mlp='mlp'

class Embeddings(str, Enum):
    avg='avg'
    sum='sum'

class ProbingJobInfo(BaseModel):
    status: JobStatus
    data: Dict
    logs: LogRecord

class UDParserStartRequest(BaseModel):
    train_path: Union[str, None] = ""
    test_path: Union[str, None] = ""
    val_path: Union[str, None] = ""
    

class UDParserStartResponse(BaseModel):
    task_id: str
    status: JobStatus
    logs: List[LogRecord]
    
class UDParserStatusRequest(BaseModel):
    task_id: str

class UDParserStatusResponse(BaseModel):
    status: str
    logs: List[LogRecord]


class ProbingFrameworkStartRequest(BaseModel):
    task_id: str = "same as was received from parser"
    model_name: HFModel
    probing_type: ProbingType
    embeddings: Embeddings
    truncation: bool
    qualification: str = "что это за поле?"
    iteration: int
    language: List[str]
    category: List[str]
    hidden_size: int
    dropout: float

class ProbingFrameworkStatusRequest(BaseModel):
    task_id: str = "same as was received from parser"


class ProbingFrameworkStartResponse(BaseModel):
    task_id: str
    status: JobStatus
    logs: List[LogRecord]

class ProbingFrameworkStatusResponse(BaseModel):
    statuses: List[ProbingJobInfo]


@app.post("/ud-parser-start", response_model=UDParserStartResponse)
def ud_parser_start(body: UDParserStartRequest):
    result = None#UDParserStartResponse()
    return result

@app.post("/ud-parser-status", response_model=UDParserStatusResponse)
def ud_parser_status(body: UDParserStatusRequest):
    result = None#UDParserStatusResponse()
    return result

@app.post("/probing-start", response_model=ProbingFrameworkStartResponse)
def probing_start(body: ProbingFrameworkStartRequest):
    result = None#ProbingFrameworkStartResponse()
    return result

@app.post("/probing-status", response_model=ProbingFrameworkStatusResponse)
def probing_status(body: ProbingFrameworkStatusRequest):
    result = None#ProbingFrameworkStartRequest()
    return result
