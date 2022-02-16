from enum import Enum
from typing import List, Optional, Tuple
import os
from tqdm import tqdm

from torch.utils.data import DataLoader
from torch.utils.data import BatchSampler

from probing.classifier import LogReg, MLP
from probing.data_former import DataFormer, EncodeLoader
from probing.encoder import TransformersLoader


class ProbingPipeline:
    def __init__(
        self,
        probing_type: Enum,
        hf_model_name: Enum,
        device: Optional[Enum] = None,
        classifier_name: Enum = "logreg",
        num_classes: int = 2,
        dropout_rate: float = 0.2,
        num_hidden: int = 256,
        batch_size: Optional[int] = 128,
        shuffle: bool = False,
    ):
        self.hf_model_name = hf_model_name
        self.probing_type = probing_type
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.num_hidden = num_hidden

        self.transformer_model = TransformersLoader(hf_model_name, device)
        self.classifier = self.get_classifier(classifier_name)

        self.embed_dim = self.transformer_model.config.hidden_size
    
    def get_classifier(self, classifier_name: Enum):
        if classifier_name == "logreg":
            return LogReg(
                input_dim = self.embed_dim,
                num_classes = self.num_classes
                )
        elif classifier_name == 'mlp':
            return MLP(
                input_dim = self.embed_dim,
                num_classes = self.num_classes,
                num_hidden =  self.num_hidden,
                dropout_rate = self.dropout_rate
            )

    def run(
        self,
        probe_task: Enum,
        path_to_task_file: Optional[os.PathLike]=None
    ):
        print(f'Task in progress: {probe_task}')
        task_dataset_dict = DataFormer(probe_task, path_to_task_file).samples
        encode_func =  self.transformer_model.encode_text
        train_loader = EncodeLoader(task_dataset_dict["tr"], encode_func, self.batch_size).dataset
        val_loader = EncodeLoader(task_dataset_dict["va"], encode_func, self.batch_size).dataset
        test_loader = EncodeLoader(task_dataset_dict["te"], encode_func, self.batch_size).dataset


