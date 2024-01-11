import os
import typing
from collections import Counter, defaultdict
from typing import DefaultDict, Dict, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from probing.types import UDProbingTaskName
from probing.utils import get_probe_task_path


class TextFormer:
    def __init__(
        self,
        probe_task: Union[UDProbingTaskName, str],
        data_path: Optional[os.PathLike] = None,
        sep: str = "\t",
        shuffle: bool = True,
    ):
        self.probe_task = probe_task
        self.shuffle = shuffle
        self.data_path = get_probe_task_path(probe_task, data_path)

        self.samples, self.unique_labels, self.num_words = self.form_data(sep=sep)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    @property
    def ratio_by_classes(self) -> Dict[str, Dict[str, int]]:
        ratio_by_classes = {}
        for class_name in self.samples:
            class_labels_all = [i[1] for i in self.samples[class_name]]
            dict_ratio_sorted = dict(sorted(dict(Counter(class_labels_all)).items()))
            ratio_by_classes[class_name] = dict_ratio_sorted
        return ratio_by_classes

    @typing.no_type_check
    def form_data(
        self, sep: str = "\t"
    ) -> Tuple[DefaultDict[str, np.ndarray], Set[str]]:
        samples_dict = defaultdict(list)
        unique_labels = set()
        dataset = pd.read_csv(self.data_path, sep=sep, header=None, dtype=str)
        for _, (stage, label, text, word_indices) in dataset.iterrows():
            num_words = len(word_indices.split(","))
            samples_dict[stage].append((text, label, word_indices))
            unique_labels.add(label)

        if self.shuffle:
            samples_dict = {
                k: np.random.permutation(v) for k, v in samples_dict.items()
            }
        else:
            samples_dict = {k: np.array(v) for k, v in samples_dict.items()}
        return samples_dict, unique_labels, num_words


class EncodedVectorFormer(Dataset):
    def __init__(self, text_vectors: torch.Tensor, label_vectors: torch.Tensor):
        self.label_vectors = label_vectors
        self.text_vectors = text_vectors

    def __len__(self):
        return len(self.label_vectors)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        label = self.label_vectors[idx]
        text = self.text_vectors[idx]
        sample = (text, label)
        return sample


class TokenizedVectorFormer(Dataset):
    def __init__(self, data: Dict[str, torch.Tensor]):
        self.input_ids = data["input_ids"]
        self.attention_masks = data["attention_mask"]
        self.labels = data["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        input_ids = self.input_ids[idx]
        attention_mask = self.attention_masks[idx]
        labels = self.labels[idx]
        sample = (input_ids, attention_mask, labels)
        return sample
