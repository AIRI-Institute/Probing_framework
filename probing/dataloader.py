from enum import Enum
import os
import numpy as np


class DataFormer:
    def __init__(self, probe_task: Enum, data_path: os.PathLike, batch_size: int=None):
        self.probe_task = probe_task
        self.data_path = data_path
        if batch_size is None:
            self.chunks = False
        else:
            self.chunks = True
        self.batch_size = batch_size
    
    def form_data(self):
        samples = []
        f = open(self.data_path)
        for i, line in enumerate(list(f)):
            subset, label, sentence = line.strip().split("\t")
            samples += [(i, subset, label, sentence)]

        if self.chunks:
            return [samples[x:x+10] for x in range(0, len(samples), self.batch_size)]
        return samples
