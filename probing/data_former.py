from enum import Enum
from typing import Tuple, Dict, Optional, List, Union, Callable, Set, DefaultDict
import os
from torch.utils.data import Dataset
import torch
import numpy as np
from collections import defaultdict

from probing.utils import get_probe_task_path


class TextFormer:
    def __init__(
        self,
        probe_task: Union[Enum, str],
        data_path: Optional[os.PathLike] = None,
        shuffle: bool = True
    ):
        self.probe_task = probe_task
        self.shuffle = shuffle
        self.data_path = get_probe_task_path(probe_task, data_path)

        self.samples, self.unique_labels = self.form_data()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def form_data(self) -> Tuple[DefaultDict[Enum, np.ndarray], Set[str]]:
        samples_dict = defaultdict(list)
        unique_labels = set()
        f = open(self.data_path)
        for line in list(f):
            stage, label, text = line.strip().split("\t")
            samples_dict[stage].append((text, label))
            unique_labels.add(label)

        if self.shuffle:
            samples_dict = {k: np.random.permutation(v) for k, v in samples_dict.items()}
        else:
            samples_dict = {k: np.array(v) for k, v in samples_dict.items()}
        return samples_dict, unique_labels


class EncodedVectorFormer(Dataset):
    def __init__(self, text_vectors, label_vectors):
        self.label_vectors = label_vectors
        self.text_vectors = text_vectors

    def __len__(self):
        return len(self.label_vectors)

    def __getitem__(self, idx):
        label = self.label_vectors[idx]
        text = self.text_vectors[idx]
        sample = (text, label)
        return sample


class TokenizedVectorFormer(Dataset):
    def __init__(self, data: Dict[Enum, torch.tensor]):
        self.input_ids = data['input_ids']
        self.attention_masks = data['attention_mask']
        self.labels = data['labels']

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        input_ids = self.input_ids[idx]
        attention_mask = self.attention_masks[idx]
        labels = self.labels[idx]
        sample = (input_ids, attention_mask, labels)
        return sample
