from time import time
from enum import Enum
from typing import Optional, Callable, Union, List, Tuple
import os
from tqdm.notebook import trange
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from probing.classifier import LogReg, MLP
from probing.data_former import TextFormer, EncodeLoader
from probing.encoder import TransformersLoader
from probing.metric import Metric
from probing.utils import save_log, get_ratio_by_classes, lang_category_extraction, ProbingLog


class ProbingPipeline:
    def __init__(
        self,
        hf_model_name: Optional[Enum] = None,
        probing_type: Optional[Enum] = "layer",
        device: Optional[Enum] = None,
        classifier_name: Enum = "logreg",
        metric_names: Union[Enum, List[Enum]] = "f1",
        embedding_type: Enum = "cls",
        encode_batch_size: Optional[int] = 64,
        probing_batch_size: Optional[int] = 64,
        dropout_rate: float = 0.2,
        hidden_size: int = 256,
        shuffle: bool = True,
        truncation: bool = False
    ):
        self.hf_model_name = hf_model_name
        self.probing_type = probing_type
        self.encode_batch_size = encode_batch_size
        self.probing_batch_size = probing_batch_size
        self.shuffle = shuffle
        self.dropout_rate = dropout_rate
        self.hidden_size = hidden_size
        self.classifier_name = classifier_name
        self.metric_names = metric_names
        self.embedding_type = embedding_type

        self.metrics = Metric(metric_names)
        self.transformer_model = TransformersLoader(
            model_name = hf_model_name,
            device = device,
            truncation = truncation
            )

    def get_classifier(
        self,
        classifier_name: Enum,
        num_classes: int,
        embed_dim: int
    ) -> Callable:
        if classifier_name == "logreg":
            return LogReg(
                input_dim = embed_dim,
                num_classes = num_classes
                )
        elif classifier_name == "mlp":
            return MLP(
                input_dim = embed_dim,
                num_classes = num_classes,
                hidden_size =  self.hidden_size,
                dropout_rate = self.dropout_rate
                )
        else:
            raise NotImplementedError(f"Unknown classifier: {classifier_name}")

    def train(
        self,
        train_loader: DataLoader,
        layer: int
    ) -> float:
        epoch_train_losses = []
        self.classifier.train()
        for x, y in train_loader:
            x = x.permute(1,0,2)
            x = torch.squeeze(x[layer], 0).to(self.transformer_model.device).float()
            y = y.to(self.transformer_model.device)

            self.classifier.zero_grad(set_to_none=True)
            prediction = self.classifier(x)
            loss = self.criterion(prediction, y)
            epoch_train_losses.append(loss.item())
            
            loss.backward()
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()

        epoch_loss = np.mean(epoch_train_losses)
        return epoch_loss
    
    def evaluate(
        self,
        dataloader: DataLoader,
        layer: int,
        save_checkpoints: bool = False
    ) -> Tuple[List[float], List[float]]:
        epoch_losses = []
        epoch_predictions = []
        epoch_true_labels = []

        self.classifier.eval()
        with torch.no_grad():
            for x, y in dataloader:
                x = x.permute(1,0,2)
                x = torch.squeeze(x[layer], 0).to(self.transformer_model.device).float()
                y = y.to(self.transformer_model.device)

                prediction = self.classifier(x)
                loss = self.criterion(prediction, y)
                epoch_losses.append(loss.item())

                if save_checkpoints:
                    raise NotImplementedError()

                epoch_predictions += prediction.data.max(1).indices.cpu()
                epoch_true_labels += y.cpu()

        epoch_metric_score = self.metrics(epoch_predictions, epoch_true_labels)
        epoch_loss = np.mean(epoch_losses)
        return epoch_loss, epoch_metric_score

    def run(
        self,
        probe_task: Union[Enum, str],
        path_to_task_file: Optional[os.PathLike] = None,
        train_epochs: int = 10,
        is_scheduler: bool = False,
        save_checkpoints: bool = False,
        verbose: bool = True
    ) -> None:
        num_layers = self.transformer_model.config.num_hidden_layers
        task_data = TextFormer(probe_task, path_to_task_file)
        task_dataset, num_classes = task_data.samples, task_data.num_classes
        path_to_file_for_probing = task_data.data_path
        task_language, task_category = lang_category_extraction(path_to_file_for_probing)

        self.log_info = ProbingLog()
        self.log_info['params']['probing_task'] = probe_task
        self.log_info['params']['file_path'] = path_to_file_for_probing
        self.log_info['params']['task_language'] = task_language
        self.log_info['params']['task_category'] = task_category
        self.log_info['params']['probing_type'] = self.probing_type
        self.log_info['params']['batch_size'] = self.encode_batch_size
        self.log_info['params']['hf_model_name'] = self.transformer_model.config._name_or_path
        self.log_info['params']['classifier_name'] = self.classifier_name
        self.log_info['params']['metric_names'] = self.metric_names
        self.log_info['params']['original_classes_ratio'] = get_ratio_by_classes(task_dataset)

        if verbose:
            print('=' * 100)
            print(f'Task in progress: {probe_task}\nPath to data: {path_to_file_for_probing}')

        start_time = time()
        encode_func =  lambda x: self.transformer_model.encode_text(x, self.embedding_type)
        probing_loader = EncodeLoader(
            encode_func = encode_func,
            encode_batch_size = self.encode_batch_size,
            probing_batch_size = self.probing_batch_size,
            shuffle = self.shuffle
            )
        tr_dataset = probing_loader(task_dataset["tr"])
        self.log_info['params']['encoded_labels'] = probing_loader.encoded_labels_dict
        tr_dataset = len(tr_dataset)
        val_dataset = probing_loader(task_dataset["va"])
        te_dataset = probing_loader(task_dataset["te"])

        probing_iter_range = trange(num_layers, desc="Probing by layers") if verbose else range(num_layers)
        self.log_info['results']['elapsed_time(sec)'] = 0
        for layer in probing_iter_range:
            self.classifier = self.get_classifier(
                self.classifier_name,
                num_classes,
                self.transformer_model.config.hidden_size
                ).to(self.transformer_model.device)
            # self.criterion = torch.nn.CrossEntropyLoss(
            #     weight=torch.Tensor(list(self.log_info['params']['original_classes_ratio']['tr'].values()))
            #     ).to(self.transformer_model.device)
            self.criterion = torch.nn.CrossEntropyLoss().to(self.transformer_model.device)
            self.optimizer = AdamW(self.classifier.parameters())
            
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=2000,
                num_training_steps=len(tr_dataset) // train_epochs
                ) if is_scheduler else None

            for epoch in range(train_epochs):
                epoch_train_loss = self.train(tr_dataset, layer)

                epoch_val_loss, epoch_val_score = self.evaluate(val_dataset, layer, save_checkpoints)

                self.log_info['results']['train_loss'].add(layer, epoch_train_loss)
                self.log_info['results']['val_loss'].add(layer, epoch_val_loss)

                for m in self.metric_names:
                    self.log_info['results']['val_score'][m].add(layer, epoch_val_score[m])

            _, epoch_test_score = self.evaluate(te_dataset, layer, save_checkpoints)

            for m in self.metric_names:
                self.log_info['results']['test_score'][m].add(layer, epoch_test_score[m])
        
        del tr_dataset
        self.log_info['results']['elapsed_time(sec)'] = time() - start_time
        output_path = save_log(self.log_info, probe_task)
        if verbose:
            print(f"Experiments were saved in the folder: {output_path}")
