from enum import Enum
from typing import List, Optional
import os
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from probing.classifier import LogReg, MLP
from probing.dataloader import DataFormer
from probing.utils import TransformersLoader


class ProbingPipeline:
    def __init__(
        self,
        probing_type: Enum,
        hf_model_name: Enum,
        device: Optional[Enum] = None,
        classifier: Enum = 'logreg',
        batch_size: Optional[int] = 128
    ):
        self.hf_model_name = hf_model_name
        self.classifier = classifier
        self.probing_type = probing_type
        self.batch_size = batch_size

        self.transformer_model = TransformersLoader(hf_model_name, device)

    def run(self, probe_task: Enum, path_to_data: Optional[os.PathLike]):
        print(f'Task in progress: {probe_task}')
        task_dataset = DataFormer(probe_task, path_to_data, self.batch_size)

        self.iterator = DataLoader(
            task_dataset, batch_size=self.batch_size, shuffle=False 
        )

        for indices, subset, label, sentences in tqdm(self.iterator):
            encoded_batch = self.transformer_model.tokenizer(
                list(sentences),
                add_special_tokens=True,
                padding="longest",
                return_tensors="pt",
            )
            input_ids = encoded_batch["input_ids"].to(self.device)
            attention_mask = encoded_batch["attention_mask"].to(self.device)
            with torch.no_grad():
                model_outputs = self.transformer_model.model(
                        input_ids, attention_mask, return_dict=True
                )
                model_outputs = (
                    model_outputs["hidden_states"]
                    if "hidden_states" in model_outputs
                    else model_outputs["encoder_hidden_states"]
                )
                for layer_i in range(self.transformer_model.config.num_hidden_layers):
                    pass
