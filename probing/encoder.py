import gc
import logging
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer

from probing.data_former import EncodedVectorFormer, TokenizedVectorFormer
from probing.utils import exclude_rows


class TransformersLoader:
    def __init__(
        self,
        model_name: Optional[Enum] = None,
        device: Optional[Enum] = None,
        truncation: bool = False,
        padding: Enum = "longest",
        return_tensors: Enum = "pt",
        add_special_tokens: bool = True,
        return_dict: bool = True,
        output_hidden_states: bool = True,
        output_attentions: bool = True
    ):
        self.config = AutoConfig.from_pretrained(
            model_name, output_hidden_states=output_hidden_states, 
            output_attentions=output_attentions
            ) if model_name else None
        self.model = AutoModel.from_pretrained(
            model_name, config=self.config
            ) if model_name else None
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, config=self.config
            ) if model_name else None

        self.cache = {}
        self.truncation = truncation 
        self.padding = padding
        self.return_tensors = return_tensors
        self.add_special_tokens = add_special_tokens
        self.return_dict = return_dict
        self.device = device
        self.init_device()

    def init_device(self):
        """
        To define the device in case it is not passed. This device is used for probing
        computational.

        If a device is None or CPU, it is trying to detect the model's device.
        For such a big model laying on several GPUs, you must pass the device directly
        to prevent `CUDA out-of-memory error.
        """
        if self.model:
            model_device = self.model.device
            if self.device and model_device.type == "cpu":
                self.model.to(torch.device(self.device))
            elif self.device is None and model_device.type != "cpu":
                self.device = model_device
            elif torch.cuda.is_available():
                self.model.cuda()
                self.device = self.model.device
            else:
                self.device = "cpu"
                self.model.to(torch.device(self.device))
        else:
            self.device = None

    def check_cache_ids(self, input_ids: torch.Tensor) -> Tuple[List[int], List[int]]:
        in_cache_ids = []
        out_cache_ids = []
        for i, element in enumerate(input_ids):
            text = self.tokenizer.decode(element)
            if text in self.cache:
                in_cache_ids.append(i)
            else:
                out_cache_ids.append(i)
        return in_cache_ids, out_cache_ids

    def add_to_cache(self, input_ids_new: torch.Tensor, model_output_tensors_new: torch.Tensor) -> None:
        for input_ids, out_cache_tensor in zip(input_ids_new, model_output_tensors_new):
            input_ids_unpad = input_ids #input_ids[input_ids != self.tokenizer.pad_token_id]
            decoded_text = self.tokenizer.decode(input_ids_unpad)
            self.cache[decoded_text] = torch.unsqueeze(out_cache_tensor, 0)
    
    def get_from_cache(self, input_ids_cached: torch.Tensor) -> List[torch.Tensor]:
        cached_tensors_list = []
        for input_ids in input_ids_cached:
            input_ids_unpad = input_ids #input_ids[input_ids != self.tokenizer.pad_token_id]
            decoded_text = self.tokenizer.decode(input_ids_unpad)
            cached_tensors_list.append(self.cache[decoded_text])
        return cached_tensors_list

    def tokenize_text(self, text: Union[str, List[str]]) -> torch.Tensor:
        tokenized_text = self.tokenizer(
            text,
            padding=self.padding,
            return_tensors=self.return_tensors,
            add_special_tokens = self.add_special_tokens,
            truncation = self.truncation
        )
        return tokenized_text

    def _fix_tokenized_tensors(self, tokenized_text: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        input_ids = tokenized_text["input_ids"]
        attention_mask = tokenized_text["attention_mask"]
        row_ids_to_exclude = []
        if not self.truncation and input_ids.size()[1] > self.tokenizer.model_max_length:
            pad_token_id = self.tokenizer.pad_token_id
            if self.tokenizer.padding_side == "left":
                row_ids_to_exclude = torch.where(input_ids[:, 0] != pad_token_id)
            else:
                row_ids_to_exclude = torch.where(input_ids[:, self.tokenizer.model_max_length - 1] != pad_token_id)
            if isinstance(row_ids_to_exclude, tuple):
                row_ids_to_exclude = row_ids_to_exclude[0]

            input_ids = exclude_rows(input_ids, row_ids_to_exclude)[:, :self.tokenizer.model_max_length]
            attention_mask = exclude_rows(attention_mask, row_ids_to_exclude)[:, :self.tokenizer.model_max_length]
            row_ids_to_exclude = row_ids_to_exclude.tolist()
        return input_ids, attention_mask, row_ids_to_exclude

    def _get_embeddings_by_layers(self, model_outputs: Tuple[torch.Tensor], embedding_type: Enum) -> List[torch.Tensor]:
        layers_outputs = []
        for output in model_outputs[1:]:
            if embedding_type == "cls":
                sent_vector = output[:, 0, :]
            elif embedding_type == "sum":
                sent_vector = torch.sum(output, dim=1)
            elif embedding_type == "avg":
                sent_vector = torch.mean(output, dim=1)
            else:
                raise NotImplementedError(
                    f'Unknown type of embedding\'s aggregation: {embedding_type}'
                    )
            layers_outputs.append(sent_vector)
        return layers_outputs

    def get_tokenized_datasets(self, task_dataset: Dict[Enum, np.ndarray]) -> Tuple[Dict[Enum, TokenizedVectorFormer], Dict[Enum, int]]:
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

            # this step is necessary because we've added some tokens (pad_token) to the embeddings
            # otherwise the tokenizer and model tensors won't match up
            self.model.resize_token_embeddings(len(self.tokenizer))

        if self.device is None or self.model.device.type == "cpu":
            self.init_device()

        encoded_stage_data_dict = {}
        encoded_stage_labels_dict = {}

        tr_text_label_data = task_dataset["tr"]
        va_text_label_data = task_dataset["va"]
        te_text_label_data = task_dataset["te"]

        for stage, text_label_data in zip(
            ["tr", "va", "te"],
            [tr_text_label_data, va_text_label_data, te_text_label_data]
        ):
            tokenized_text = self.tokenize_text(text_label_data[:,0].tolist())

            if stage == "tr":
                label_encoder = LabelEncoder()
                label_encoder.fit(text_label_data[:,1])

            input_ids, attention_mask, row_ids_to_exclude = self._fix_tokenized_tensors(tokenized_text)
            labels = label_encoder.transform(text_label_data[:,1])
            fixed_labels = exclude_rows(torch.tensor(labels), row_ids_to_exclude).view(-1).tolist()

            if row_ids_to_exclude:
                logging.warning(f"Since you decided not to truncate long sentences, {len(row_ids_to_exclude)} sample(s) were excluded")

            encoded_stage_labels_dict[stage] = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
            encoded_stage_data_dict[stage] = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": fixed_labels}

        tokenized_tr_data = TokenizedVectorFormer(encoded_stage_data_dict["tr"])
        tokenized_va_data = TokenizedVectorFormer(encoded_stage_data_dict["va"])
        tokenized_te_data = TokenizedVectorFormer(encoded_stage_data_dict["te"])
        tokenized_datasets = {"tr": tokenized_tr_data,"va": tokenized_va_data,"te": tokenized_te_data}
        return tokenized_datasets, encoded_stage_labels_dict

    def model_layers_forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, embedding_type: Enum) -> List[torch.Tensor]:
        if hasattr(self.model, "encoder") and hasattr(self.model, "decoder"):
            # In case of encoder-decoder model, for embeddings we use only encoder 
            model_outputs = self.model.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=self.return_dict
                )
        else:
            # In case of the models with decoder or encoder only
            model_outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=self.return_dict
                )

        model_outputs = (
            model_outputs["hidden_states"]
            if "hidden_states" in model_outputs
            else model_outputs["encoder_hidden_states"]
        )
        layers_outputs = self._get_embeddings_by_layers(model_outputs, embedding_type=embedding_type)
        return layers_outputs

    def encode_data(
        self,
        data: DataLoader,
        stage: Enum,
        embedding_type: Enum,
        verbose: bool,
        do_control_task: bool = False
    ) -> EncodedVectorFormer:
        encoded_text_tensors = []
        label_vectors = []
        
        self.model.eval()
        torch.cuda.empty_cache()
        gc.collect()
        with torch.no_grad():
            iter_data = tqdm(data, total = len(data), desc=f"Data encoding {stage}") if verbose else data
            for batch_input_ids, batch_attention_mask, batch_labels in iter_data:
                in_cache_ids, out_cache_ids = self.check_cache_ids(batch_input_ids)
                input_ids_out = batch_input_ids[out_cache_ids].to(self.device, non_blocking=True)
                input_ids_in = batch_input_ids[in_cache_ids].to(self.device, non_blocking=True)

                attention_mask_out = batch_attention_mask[out_cache_ids].to(self.device, non_blocking=True)
                # attention_mask_in = batch_attention_mask[in_cache_ids].to(self.device)

                labels_out = batch_labels[out_cache_ids]
                labels_in = batch_labels[in_cache_ids]
                labels = torch.cat((labels_in, labels_out))
                
                if len(input_ids_out):
                    layers_outputs = self.model_layers_forward(input_ids_out, attention_mask_out, embedding_type)
                    encoded_batch_text_tensor = torch.stack(layers_outputs)
                    out_cache_encoded_batch_vectors = encoded_batch_text_tensor.permute(1,0,2)
                    
                    # add to cache
                    self.add_to_cache(input_ids_out, out_cache_encoded_batch_vectors.cpu())
                else:
                    out_cache_encoded_batch_vectors = []
                    
                # get from cache
                cached_tensors_list = self.get_from_cache(input_ids_in)

                if len(cached_tensors_list):
                    cached_tensors = torch.cat(cached_tensors_list).to(self.device, non_blocking=True)
                    final_tensor = torch.cat((out_cache_encoded_batch_vectors, cached_tensors)) if len(out_cache_encoded_batch_vectors) else cached_tensors
                else:
                    final_tensor = out_cache_encoded_batch_vectors

                encoded_text_tensors.append(final_tensor)
                label_vectors.append(labels)

        encoded_text_tensors = torch.cat(encoded_text_tensors, dim=0)
        label_vectors = torch.cat(label_vectors, dim=0)
        if do_control_task:
            idx = torch.randperm(label_vectors.shape[0])
            label_vectors = label_vectors[:, idx]
        probe_dataset = EncodedVectorFormer(encoded_text_tensors, label_vectors)
        return probe_dataset

    def get_encoded_dataloaders(
        self,
        task_dataset: Dict[Enum, np.ndarray],
        encoding_batch_size: int = 64,
        classifier_batch_size: int = 64,
        shuffle: bool = True,
        embedding_type: Enum = "cls",
        verbose: bool = True,
        do_control_task: bool = False
    ) -> Tuple[Dict[Enum, DataLoader], Dict[Enum, int]]:
        tokenized_datasets, encoded_labels = self.get_tokenized_datasets(task_dataset)
        tr_dataloader_tokenized = DataLoader(tokenized_datasets["tr"], batch_size=encoding_batch_size)
        va_dataloader_tokenized = DataLoader(tokenized_datasets["va"], batch_size=encoding_batch_size)
        te_dataloader_tokenized = DataLoader(tokenized_datasets["te"], batch_size=encoding_batch_size)

        tr_tokenized = self.encode_data(tr_dataloader_tokenized, "train", embedding_type, verbose, do_control_task=do_control_task)
        va_tokenized = self.encode_data(va_dataloader_tokenized, "val", embedding_type, verbose, do_control_task=do_control_task)
        te_tokenized = self.encode_data(te_dataloader_tokenized, "test", embedding_type, verbose, do_control_task=do_control_task)

        tr_dataloader_encoded = DataLoader(tr_tokenized, batch_size=classifier_batch_size, shuffle=shuffle)
        va_dataloader_encoded = DataLoader(va_tokenized, batch_size=classifier_batch_size, shuffle=shuffle)
        te_dataloader_encoded = DataLoader(te_tokenized, batch_size=classifier_batch_size, shuffle=shuffle)
        return {"tr": tr_dataloader_encoded, "va": va_dataloader_encoded, "te": te_dataloader_encoded}, encoded_labels
