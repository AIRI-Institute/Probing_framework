from enum import Enum
from typing import Tuple, Dict, Optional, List, Union, Callable
import os
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset, BatchSampler
import torch
import logging
import numpy as np
from sklearn import preprocessing

from probing.utils import get_probe_task_path, exclude_rows


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

        self.samples, self.unique_labels = self.__form_data()
        self.num_classes = len(self.unique_labels)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def __form_data(self) -> Dict[Enum, Tuple[Enum, str]]:
        samples_dict = {}
        unique_labels = []
        f = open(self.data_path)
        for line in list(f):
            data_type, label, text = line.strip().split("\t")
            if data_type not in samples_dict:
                samples_dict[data_type] = []
            samples_dict[data_type].append((text, label))
            if label not in unique_labels:
                unique_labels.append(label)

        if self.shuffle:
            samples_dict = {k: np.random.permutation(v) for k, v in samples_dict.items()}
        return samples_dict, unique_labels


class VectorFormer(Dataset):
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


class EncodeLoader:
    def __init__(
        self,
        encode_func: Callable,
        encode_batch_size: int = 64,
        probing_batch_size: int = 64,
        drop_last: bool = False,
        shuffle: bool = True
    ):  
        self.encode_func = encode_func
        self.encode_batch_size = encode_batch_size
        self.probing_batch_size = probing_batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle

    def __call__(self, list_texts_labels: List[Tuple[str, Enum]]) -> DataLoader:
        return self.__form_dataloader(list_texts_labels)

    def __len__(self):
        return len(self.dataset)
    
    def __label_encoder(self, array: List[str]) -> List[int]:
        le = preprocessing.LabelEncoder()
        le.fit(array)
        self.encoded_labels_dict = dict(zip(le.classes_, le.transform(le.classes_)))
        if len(self.encoded_labels_dict) < 2:
            logging.warning("Provided data contains only one class")
        return le.transform(array)

    def __batch_sampler(self, list_data) -> BatchSampler:
        return BatchSampler(
            list_data,
            batch_size = self.encode_batch_size,
            drop_last = self.drop_last
        )

    def __get_sampled_data(self, list_texts_labels: List[Tuple[str, Enum]]) -> Tuple[List[str], List[int]]:
        texts, labels = [], []
        for sample in list_texts_labels:
            texts.append(sample[0])
            labels.append(sample[1])
        
        encoded_labels = self.__label_encoder(labels)
        sampled_texts = self.__batch_sampler(texts)
        sampled_labels = self.__batch_sampler(encoded_labels)
        return sampled_texts, sampled_labels

    def __form_dataloader(self, list_texts_labels: List[Tuple[str, Enum]]) -> DataLoader:
        sampled_texts, sampled_labels = self.__get_sampled_data(list_texts_labels)
        text_vectors = []
        label_vectors = []
        all_excluded_rows = []
        for batch_text, batch_label in tqdm(
            zip(sampled_texts, sampled_labels), 
            total = len(sampled_texts),
            desc='Data encoding'
        ):
            encoded_batch_text, row_ids_to_exclude = self.encode_func(batch_text)
            encoded_batch_text_permuted = encoded_batch_text.permute(1,0,2)
            fixed_labels = exclude_rows(torch.tensor(batch_label), row_ids_to_exclude).view(-1).tolist()
            all_excluded_rows.extend(row_ids_to_exclude)

            text_vectors.append(encoded_batch_text_permuted)
            label_vectors.extend(fixed_labels)

        if all_excluded_rows:
            logging.warning(f"Since you decided not to truncate long sentences, {len(all_excluded_rows)} samples were excluded")
        
        vectors_tensor = torch.cat(text_vectors, dim=0)
        probe_dataset = VectorFormer(vectors_tensor, label_vectors)
        return DataLoader(
            dataset = probe_dataset,
            batch_size = self.probing_batch_size,
            pin_memory = True
        )
