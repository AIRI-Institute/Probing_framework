import logging as info_logging
import typing
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers.utils import logging

from probing.data_former import EncodedVectorFormer, TokenizedVectorFormer
from probing.types import AggregationName, AggregationType
from probing.utils import clear_memory

logging.set_verbosity_warning()
logger = logging.get_logger("probing")


class TransformersLoader:
    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        truncation: bool = False,
        padding: str = "longest",
        return_tensors: str = "pt",
        add_special_tokens: bool = True,
        return_dict: bool = True,
        output_hidden_states: bool = True,
        output_attentions: bool = True,
        model_max_length: int = 512,
    ):
        self.config = (
            AutoConfig.from_pretrained(
                model_name,
                output_hidden_states=output_hidden_states,
                output_attentions=output_attentions,
            )
            if model_name
            else None
        )
        self.model = (
            AutoModel.from_pretrained(model_name, config=self.config)
            if model_name
            else None
        )
        self.tokenizer = (
            AutoTokenizer.from_pretrained(model_name, config=self.config)
            if model_name
            else None
        )

        self.cache: Dict[str, torch.Tensor] = {}
        self.truncation = truncation
        self.padding = padding
        self.return_tensors = return_tensors
        self.add_special_tokens = add_special_tokens
        self.return_dict = return_dict
        self.device = device
        self.model_max_length = model_max_length

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

    def add_to_cache(
        self, input_ids_new: torch.Tensor, model_output_tensors_new: torch.Tensor
    ) -> None:
        for input_ids, out_cache_tensor in zip(input_ids_new, model_output_tensors_new):
            input_ids_unpad = input_ids
            decoded_text = self.tokenizer.decode(input_ids_unpad)
            self.cache[decoded_text] = torch.unsqueeze(out_cache_tensor, 0)

    def get_from_cache(self, input_ids_cached: torch.Tensor) -> List[torch.Tensor]:
        cached_tensors_list = []
        for input_ids in input_ids_cached:
            input_ids_unpad = input_ids
            decoded_text = self.tokenizer.decode(input_ids_unpad)
            cached_tensors_list.append(self.cache[decoded_text])
        return cached_tensors_list

    def tokenize_text(self, text: Union[str, List[str]]) -> torch.Tensor:
        tokenized_text = self.tokenizer(
            text,
            padding=self.padding,
            return_tensors=self.return_tensors,
            add_special_tokens=self.add_special_tokens,
            truncation=self.truncation,
        )
        return tokenized_text

    def exclude_rows(
        self, tensor: torch.Tensor, rows_to_exclude: torch.Tensor
    ) -> torch.Tensor:
        if len(tensor.size()) == 1:
            tensor = tensor.view(-1, 1)

        tensor_shape = tensor.size()
        assert len(tensor_shape) == 2
        tensor = tensor.view(*tensor_shape, 1)

        mask = torch.ones(tensor_shape, dtype=torch.bool)
        mask[rows_to_exclude, :] = False
        new_num_rows = tensor_shape[0] - len(rows_to_exclude)
        if new_num_rows == 0:
            info_logging.warning(
                "All samples were excluded due to long sentences truncation"
            )
            return tensor[mask]
        output = tensor[mask].view(new_num_rows, -1)
        return output

    @typing.no_type_check
    def _fix_tokenized_tensors(
        self, tokenized_text: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        input_ids = tokenized_text["input_ids"]
        attention_mask = tokenized_text["attention_mask"]
        if not self.truncation and input_ids.size()[1] > self.model_max_length:
            pad_token_id = self.tokenizer.pad_token_id
            if self.tokenizer.padding_side == "left":
                row_ids_to_exclude = torch.where(
                    input_ids[:, -self.model_max_length - 1] != pad_token_id
                )
            else:
                row_ids_to_exclude = torch.where(
                    input_ids[:, self.model_max_length - 1] != pad_token_id
                )
            if isinstance(row_ids_to_exclude, tuple):
                row_ids_to_exclude = row_ids_to_exclude[0]

            input_ids = self.exclude_rows(input_ids, row_ids_to_exclude)[
                :, : self.model_max_length
            ]
            attention_mask = self.exclude_rows(attention_mask, row_ids_to_exclude)[
                :, : self.model_max_length
            ]
            row_ids_to_exclude = row_ids_to_exclude.tolist()
        else:
            row_ids_to_exclude = []
        return input_ids, attention_mask, row_ids_to_exclude

    def _get_embeddings_by_layers(
        self,
        model_outputs: Tuple[torch.Tensor],
        aggregation_embeddings: AggregationName,
    ) -> List[torch.Tensor]:
        layers_outputs = []
        for output in model_outputs[1:]:  # type: ignore
            if aggregation_embeddings == AggregationType.cls:
                sent_vector = output[:, 0, :]  # type: ignore
            elif aggregation_embeddings == AggregationType.sum:
                sent_vector = torch.sum(output, dim=1)
            elif aggregation_embeddings == AggregationType.avg:
                sent_vector = torch.mean(output, dim=1)
            else:
                raise NotImplementedError(
                    f"Unknown type of embedding's aggregation: {aggregation_embeddings}"
                )
            layers_outputs.append(sent_vector)
        return layers_outputs

    def get_tokenized_datasets(
        self, task_dataset: Dict[Literal["tr", "va", "te"], np.ndarray]
    ) -> Dict[Literal["tr", "va", "te"], TokenizedVectorFormer]:
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

            # this step is necessary because we've added some tokens (pad_token) to the embeddings
            # otherwise the tokenizer and model tensors won't match up
            self.model.resize_token_embeddings(len(self.tokenizer))

        if self.device is None or self.model.device.type == "cpu":
            self.init_device()

        encoded_stage_data_dict = {}
        label_encoder = LabelEncoder()

        for stage in task_dataset.keys():
            stage_data = task_dataset[stage]
            tokenized_text = self.tokenize_text(stage_data[:, 0].tolist())
            numeric_labels = stage_data[:, 1]

            if stage == "tr":
                labels = label_encoder.fit_transform(numeric_labels)
                encoder_labels_mapping = dict(
                    zip(
                        label_encoder.classes_,
                        label_encoder.transform(label_encoder.classes_),
                    )
                )
            else:
                labels = label_encoder.transform(numeric_labels)

            input_ids, attention_mask, row_ids_to_exclude = self._fix_tokenized_tensors(
                tokenized_text
            )

            if row_ids_to_exclude:
                logger.warning(
                    f"Since you decided not to truncate long sentences, {len(row_ids_to_exclude)} sample(s) were excluded"
                )

            stage_data_dict = TokenizedVectorFormer(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": self.exclude_rows(
                        torch.tensor(labels), row_ids_to_exclude
                    ).view(-1),
                }
            )
            if stage == "tr":
                stage_data_dict.mapped_labels = encoder_labels_mapping  # type: ignore
            encoded_stage_data_dict[stage] = stage_data_dict

        return encoded_stage_data_dict

    def model_layers_forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        aggregation_embeddings: AggregationName,
    ) -> List[torch.Tensor]:
        if hasattr(self.model, "encoder") and hasattr(self.model, "decoder"):
            # In case of encoder-decoder model, for embeddings we use only encoder
            model_outputs = self.model.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=self.return_dict,
            )
        else:
            # In case of the models with decoder or encoder only
            model_outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=self.return_dict,
            )

        model_outputs = (
            model_outputs["hidden_states"]
            if "hidden_states" in model_outputs
            else model_outputs["encoder_hidden_states"]
        )
        layers_outputs = self._get_embeddings_by_layers(
            model_outputs, aggregation_embeddings=aggregation_embeddings
        )
        return layers_outputs

    def encode_data(
        self,
        data: DataLoader,
        stage: Literal["tr", "va", "te"],
        aggregation_embeddings: AggregationName,
        verbose: bool,
        do_control_task: bool = False,
    ) -> EncodedVectorFormer:
        encoded_text_list = []
        labels_list = []

        self.model.eval()
        clear_memory()
        with torch.no_grad():
            iter_data = (
                tqdm(data, total=len(data), desc=f"Data encoding {stage}")
                if verbose
                else data
            )
            for batch_input_ids, batch_attention_mask, batch_labels in iter_data:
                in_cache_ids, out_cache_ids = self.check_cache_ids(batch_input_ids)
                input_ids_out = batch_input_ids[out_cache_ids].to(
                    self.device, non_blocking=True
                )
                input_ids_in = batch_input_ids[in_cache_ids].to(
                    self.device, non_blocking=True
                )

                attention_mask_out = batch_attention_mask[out_cache_ids].to(
                    self.device, non_blocking=True
                )
                # attention_mask_in = batch_attention_mask[in_cache_ids].to(self.device)

                labels_out = batch_labels[out_cache_ids]
                labels_in = batch_labels[in_cache_ids]
                labels = torch.cat((labels_in, labels_out))

                if len(input_ids_out):
                    layers_outputs = self.model_layers_forward(
                        input_ids_out, attention_mask_out, aggregation_embeddings
                    )
                    encoded_batch_text_tensor = torch.stack(layers_outputs)
                    out_cache_encoded_batch_vectors = encoded_batch_text_tensor.permute(
                        1, 0, 2
                    )

                    # add to cache
                    self.add_to_cache(
                        input_ids_out, out_cache_encoded_batch_vectors.cpu()
                    )
                else:
                    out_cache_encoded_batch_vectors = torch.Tensor()

                # get from cache
                cached_tensors_list = self.get_from_cache(input_ids_in)

                if len(cached_tensors_list):
                    cached_tensors = torch.cat(cached_tensors_list).to(
                        self.device, non_blocking=True
                    )
                    final_tensor = (
                        torch.cat((out_cache_encoded_batch_vectors, cached_tensors))
                        if len(out_cache_encoded_batch_vectors)
                        else cached_tensors
                    )
                else:
                    final_tensor = out_cache_encoded_batch_vectors

                encoded_text_list.append(final_tensor)
                labels_list.append(labels)

        encoded_text_tensor = torch.cat(encoded_text_list, dim=0)
        labels_tensor = torch.cat(labels_list, dim=0)
        if do_control_task:
            idx = torch.randperm(labels_tensor.shape[0])
            labels_tensor = labels_tensor[idx]
        probe_dataset = EncodedVectorFormer(encoded_text_tensor, labels_tensor)
        return probe_dataset

    def get_encoded_dataloaders(
        self,
        task_dataset: Dict[Literal["tr", "va", "te"], np.ndarray],
        encoding_batch_size: int = 64,
        classifier_batch_size: int = 64,
        shuffle: bool = True,
        aggregation_embeddings: AggregationName = AggregationType.cls,
        verbose: bool = True,
        do_control_task: bool = False,
    ) -> Tuple[Dict[Literal["tr", "va", "te"], DataLoader], Dict[str, int]]:
        if self.tokenizer.model_max_length > self.model_max_length:
            logger.warning(
                f"In tokenizer model_max_length = {self.tokenizer.model_max_length}. Changed to {self.model_max_length} for preventing Out-Of-Memory."
            )

        tokenized_datasets = self.get_tokenized_datasets(task_dataset)
        encoded_dataloaders = {}
        for stage, _ in tokenized_datasets.items():
            stage_dataloader_tokenized = DataLoader(
                tokenized_datasets[stage], batch_size=encoding_batch_size
            )

            stage_encoded_data = self.encode_data(
                stage_dataloader_tokenized,
                stage,
                aggregation_embeddings,
                verbose,
                do_control_task=do_control_task,
            )

            encoded_dataloaders[stage] = DataLoader(
                stage_encoded_data, batch_size=classifier_batch_size, shuffle=shuffle
            )
        return encoded_dataloaders, tokenized_datasets["tr"].mapped_labels  # type: ignore
