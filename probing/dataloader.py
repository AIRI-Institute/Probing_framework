from enum import Enum
from typing import List, Tuple, Union, Optional
import os
import numpy as np

from utils import get_probe_task_path
from config import PROB_DATA_FORMAT


class DataFormer:
    def __init__(
        self,
        probe_task: Enum,
        data_path: Optional[os.PathLike] = None,
        batch_size: Optional[int] = None
    ):
        self.probe_task = probe_task
        self.data_path = get_probe_task_path(data_path)
        if batch_size is None:
            self.chunks = False
        else:
            self.chunks = True
        self.batch_size = batch_size

        self.samples = self.__form_data()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def __form_data(self) -> Union[List[PROB_DATA_FORMAT], List[List[PROB_DATA_FORMAT]]]:
        samples = []
        f = open(self.data_path)
        for i, line in enumerate(list(f)):
            subset, label, sentence = line.strip().split("\t")
            samples += [(i, subset, label, sentence)]

        if self.chunks:
            return [samples[x:x+self.batch_size] for x in range(0, len(samples), self.batch_size)]
        return samples
