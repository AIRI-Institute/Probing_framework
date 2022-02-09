from enum import Enum
from typing import List, Optional

from probing.classifier import LogReg, MLP
from probing.dataloader import DataFormer
from probing.utils import TransformersLoader


class ProbingPipeline:
    def __init__(
        self,
        hf_model_name: Enum,
        classifier: Enum,
        probing_type: Enum,
        batch_size: Optional[int] = None,
    ):
        self.hf_model_name = hf_model_name
        self.classifier = classifier
        self.probing_type = probing_type
        self.batch_size = batch_size

    def run(self, probe_task, path_to_data):
        print(f'Task in progress: {probe_task}')
        task_dataset = DataFormer(probe_task, path_to_data, self.batch_size)

