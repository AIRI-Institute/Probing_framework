from enum import Enum
from typing import Optional, List, Union
import os
import glob

from transformers import  AutoConfig, AutoModel, AutoTokenizer
import torch


def get_probe_task_path(
    probe_task_name: Enum,
    data_dir_path: Optional[str] = None
) -> os.PathLike:
    if probe_task_name.lower().startswith('ud_'):
        if data_dir_path is None:
            data_dir_path = 'data/'

        path_to_folder = os.path.join(os.getcwd(), data_dir_path, probe_task_name)
        path_to_file = glob.glob(f'{path_to_folder}*')

        if len(path_to_file) == 0:
            raise RuntimeError(f'We didn\'t find any files for the task: {probe_task_name}')
        return path_to_file[0]

    if data_dir_path is None:
        raise RuntimeError(f'You should pass a dataset path for the task: {probe_task_name}')
    return data_dir_path


class TransformersLoader:
    def __init__(
        self,
        model_name: Enum,
        device: Optional[Enum] = None
    ):
        self.config = AutoConfig.from_pretrained(
            model_name, output_hidden_states=True, output_attentions=True
        )
        self.model = AutoModel.from_pretrained(model_name, config=self.config)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, config=self.config)

        if device:
            self.device = device
            self.model.to(torch.device(self.device))
        elif torch.cuda.is_available():
            self.model.cuda()
            self.device = self.model.device
        else:
            self.device = "cpu"
            self.model.to(torch.device(self.device))

    def encode_text(self, text: Union[str, List[str], List[List[str]]]):
        encoded_text = self.tokenizer(text, padding="longest", return_tensors="pt")
        input_ids = encoded_text["input_ids"].to(self.device)
        attention_mask = encoded_text["attention_mask"].to(self.device)
        with torch.no_grad():
            model_outputs = self.model(
                    input_ids, attention_mask, return_dict=True
            )
            model_outputs = (
                model_outputs["hidden_states"]
                if "hidden_states" in model_outputs
                else model_outputs["encoder_hidden_states"]
            )
            return model_outputs
