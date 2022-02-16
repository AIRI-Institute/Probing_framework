from enum import Enum
from typing import List, Optional, Tuple
import os
from tqdm import tqdm

from torch.utils.data import DataLoader
from torch.utils.data import BatchSampler

from probing.classifier import LogReg, MLP
from probing.data_former import DataFormer, EncodeLoader
from probing.encoders import TransformersLoader


class ProbingPipeline:
    def __init__(
        self,
        probing_type: Enum,
        hf_model_name: Enum,
        device: Optional[Enum] = None,
        classifier: Enum = "logreg",
        batch_size: Optional[int] = 128,
        shuffle: bool = False,
    ):
        self.hf_model_name = hf_model_name
        self.classifier = classifier
        self.probing_type = probing_type
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.transformer_model = TransformersLoader(hf_model_name, device)

    def run(
        self,
        probe_task: Enum,
        path_to_task_file: Optional[os.PathLike]=None
    ):
        print(f'Task in progress: {probe_task}')
        task_dataset_dict = DataFormer(probe_task, path_to_task_file).samples
        encode_func =  self.transformer_model.encode_text
        train_loader = EncodeLoader(task_dataset_dict["tr"], encode_func, self.batch_size)
        val_loader = EncodeLoader(task_dataset_dict["va"], encode_func, self.batch_size)
        test_loader = EncodeLoader(task_dataset_dict["te"], encode_func, self.batch_size)


