import gc
import os
from time import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import trange
from transformers import get_linear_schedule_with_warmup
from transformers.utils import logging

from probing.classifier import MLP, LogReg, MDLLinearModel
from probing.data_former import TextFormer
from probing.encoder import TransformersLoader
from probing.metric import Metric
from probing.types import (
    AggregationName,
    AggregationType,
    ClassifierName,
    ClassifierType,
    MetricName,
    MetricType,
    ProbingName,
    ProbingType,
)
from probing.utils import KL_Loss, ProbingLog, clear_memory, lang_category_extraction

logging.set_verbosity_warning()
logger = logging.get_logger("probing")


class ProbingPipeline:
    def __init__(
        self,
        hf_model_name: Optional[str] = None,
        probing_type: Optional[ProbingName] = ProbingType.layerwise,
        device: Optional[str] = None,
        classifier_name: ClassifierName = ClassifierType.logreg,
        metric_names: Union[MetricName, List[MetricName]] = MetricType.f1,
        aggregation_embeddings: AggregationName = AggregationType.cls,
        encoding_batch_size: int = 32,
        classifier_batch_size: int = 64,
        dropout_rate: float = 0.2,
        hidden_size: int = 256,
        shuffle: bool = True,
        truncation: bool = False,
    ):
        self.hf_model_name = hf_model_name
        self.probing_type = probing_type
        self.encoding_batch_size = encoding_batch_size
        self.classifier_batch_size = classifier_batch_size
        self.shuffle = shuffle
        self.dropout_rate = dropout_rate
        self.hidden_size = hidden_size
        self.classifier_name = classifier_name
        self.aggregation_embeddings = aggregation_embeddings

        self.metric_names = (
            metric_names if isinstance(metric_names, list) else [metric_names]
        )
        self.metrics = Metric(metric_names)
        self.transformer_model = TransformersLoader(
            model_name=hf_model_name, device=device, truncation=truncation
        )
        self.criterion: Any = None

    def get_classifier(
        self, classifier_name: ClassifierName, num_classes: int, embed_dim: int
    ) -> Union[LogReg, MLP, MDLLinearModel]:
        if classifier_name == ClassifierType.logreg:
            return LogReg(input_dim=embed_dim, num_classes=num_classes)
        if classifier_name == ClassifierType.mlp:
            return MLP(
                input_dim=embed_dim,
                num_classes=num_classes,
                hidden_size=self.hidden_size,
                dropout_rate=self.dropout_rate,
            )
        if classifier_name == ClassifierType.mdl:
            return MDLLinearModel(
                input_dim=embed_dim,
                num_classes=num_classes,
            )
        raise NotImplementedError(f"Unknown classifier: {classifier_name}")

    def train(self, train_loader: DataLoader, layer: int) -> float:
        epoch_train_losses = []
        self.classifier.train()
        for i, batch in enumerate(train_loader):
            # x is already on device since it was passed through the model
            y = batch[1].to(self.transformer_model.device, non_blocking=True)

            x = batch[0].permute(1, 0, 2)
            x = torch.squeeze(x[layer], 0).float()
            x = torch.unsqueeze(x, 0) if len(x.size()) == 1 else x

            self.classifier.zero_grad(set_to_none=True)
            prediction = self.classifier(x)
            loss = self.criterion(prediction, y)
            epoch_train_losses.append(loss.item())
            loss.backward()

            if (i + 1) % 2 == 0 or (i + 1) == len(train_loader):
                self.optimizer.step()
                if self.scheduler:
                    self.scheduler.step()

        epoch_loss = np.mean(epoch_train_losses)
        return epoch_loss

    def evaluate(
        self, dataloader: DataLoader, layer: int, save_checkpoints: bool = False
    ) -> Tuple[List[float], Dict[MetricName, float]]:
        epoch_losses = []
        epoch_predictions = []
        epoch_true_labels = []

        self.classifier.eval()
        with torch.no_grad():
            for x, y in dataloader:
                # x is already on device since it was passed through the model
                y = y.to(self.transformer_model.device, non_blocking=True)

                x = x.permute(1, 0, 2)
                x = torch.squeeze(x[layer], 0).float()
                x = torch.unsqueeze(x, 0) if len(x.size()) == 1 else x

                prediction = self.classifier(x)
                loss = self.criterion(prediction, y)
                epoch_losses.append(loss.item())

                if save_checkpoints:
                    raise NotImplementedError()

                epoch_predictions += prediction.cpu().data.max(1).indices
                epoch_true_labels += y.cpu()

        epoch_metric_score = self.metrics.compute(epoch_predictions, epoch_true_labels)
        epoch_loss = np.mean(epoch_losses)
        return epoch_loss, epoch_metric_score

    def run(
        self,
        probe_task: str,
        path_to_task_file: Optional[os.PathLike] = None,
        train_epochs: int = 10,
        is_scheduler: bool = False,
        save_checkpoints: bool = False,
        verbose: bool = True,
        do_control_task: bool = False,
    ) -> None:
        task_data = TextFormer(probe_task, path_to_task_file)
        task_dataset, num_classes = task_data.samples, len(task_data.unique_labels)
        task_language, task_category = lang_category_extraction(task_data.data_path)

        log_info = ProbingLog()
        log_info["params"]["probing_task"] = probe_task
        log_info["params"]["file_path"] = task_data.data_path
        log_info["params"]["task_language"] = task_language
        log_info["params"]["task_category"] = task_category
        log_info["params"]["probing_type"] = self.probing_type
        log_info["params"]["encoding_batch_size"] = self.encoding_batch_size
        log_info["params"]["classifier_batch_size"] = self.classifier_batch_size
        log_info["params"][
            "hf_model_name"
        ] = self.transformer_model.config._name_or_path
        log_info["params"]["classifier_name"] = self.classifier_name
        log_info["params"]["metric_names"] = self.metric_names
        log_info["params"]["original_classes_ratio"] = task_data.ratio_by_classes

        if verbose:
            print(
                f"Task in progress: {probe_task}\nPath to data: {task_data.data_path}"
            )

        clear_memory()
        start_time = time()
        (
            probing_dataloaders,
            mapped_labels,
        ) = self.transformer_model.get_encoded_dataloaders(
            task_dataset,
            self.encoding_batch_size,
            self.classifier_batch_size,
            self.shuffle,
            self.aggregation_embeddings,
            verbose,
            do_control_task=do_control_task,
        )

        probing_iter_range = (
            trange(
                self.transformer_model.config.num_hidden_layers,
                desc="Probing by layers",
            )
            if verbose
            else range(self.transformer_model.config.num_hidden_layers)
        )
        log_info["params"]["tr_mapped_labels"] = mapped_labels
        log_info["results"]["elapsed_time(sec)"] = 0

        for layer in probing_iter_range:
            self.classifier = self.get_classifier(
                self.classifier_name,
                num_classes,
                self.transformer_model.config.hidden_size,
            ).to(self.transformer_model.device)

            loss_func = torch.nn.CrossEntropyLoss().to(self.transformer_model.device)
            if self.classifier == "mdl":
                self.criterion = KL_Loss(loss=loss_func)
            else:
                self.criterion = loss_func

            self.optimizer = AdamW(self.classifier.parameters())

            self.scheduler = (
                get_linear_schedule_with_warmup(
                    self.optimizer,
                    num_warmup_steps=2000,
                    num_training_steps=len(probing_dataloaders["tr"]) // train_epochs,
                )
                if is_scheduler
                else None
            )

            for epoch in range(train_epochs):
                epoch_train_loss = self.train(probing_dataloaders["tr"], layer)

                epoch_val_loss, epoch_val_score = self.evaluate(
                    probing_dataloaders["va"], layer, save_checkpoints
                )

                log_info["results"]["train_loss"].add(layer, epoch_train_loss)
                log_info["results"]["val_loss"].add(layer, epoch_val_loss)

                for m in self.metric_names:
                    log_info["results"]["val_score"][m].add(layer, epoch_val_score[m])

            _, epoch_test_score = self.evaluate(
                probing_dataloaders["te"], layer, save_checkpoints
            )

            for m in self.metric_names:
                log_info["results"]["test_score"][m].add(layer, epoch_test_score[m])

        log_info["results"]["elapsed_time(sec)"] = time() - start_time
        output_path = log_info.save_log(probe_task)
        if verbose:
            print(f"Experiments were saved in the folder: {str(output_path)}")
