from time import time
from enum import Enum
from typing import Optional, Callable, Union, List, Tuple
import os
from tqdm.notebook import trange
import numpy as np
import torch
from collections import defaultdict
from torch.utils.data import DataLoader
from torch.optim import AdamW

from probing.classifier import LogReg, MLP
from probing.data_former import DataFormer, EncodeLoader
from probing.encoder import TransformersLoader
from probing.metric import Metric
from probing.utils import save_log, get_ratio_by_classes, lang_category_extraction


class ProbingPipeline:
    def __init__(
        self,
        hf_model_name: Optional[Enum] = None,
        probing_type: Optional[Enum] = "layer",
        device: Optional[Enum] = None,
        classifier_name: Enum = "logreg",
        metric_names: Union[Enum, List[Enum]] = "accuracy",
        embedding_type: Enum = "cls",
        batch_size: Optional[int] = 64,
        dropout_rate: float = 0.2,
        hidden_size: int = 256,
        shuffle: bool = True,
        truncation: bool = False
    ):
        self.hf_model_name = hf_model_name
        self.probing_type = probing_type
        self.batch_size = batch_size
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
        self.device = self.transformer_model.device

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
                ).to(self.device)
        elif classifier_name == "mlp":
            return MLP(
                input_dim = embed_dim,
                num_classes = num_classes,
                hidden_size =  self.hidden_size,
                dropout_rate = self.dropout_rate
                ).to(self.device)
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
            x = torch.squeeze(x[layer], 0).to(self.device)
            y = torch.tensor(y).to(self.device)

            prediction = self.classifier(x)
            loss = self.criterion(prediction, y)
            epoch_train_losses.append(loss.item())
            
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

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
                x = torch.squeeze(x[layer], 0).to(self.device)
                y = torch.tensor(y).to(self.device)

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
        save_checkpoints: bool = False,
        verbose: bool = True
    ) -> None:
        num_layers = self.transformer_model.config.num_hidden_layers
        task_data = DataFormer(probe_task, path_to_task_file)
        task_dataset, num_classes = task_data.samples, task_data.num_classes
        path_to_file_for_probing = task_data.data_path
        task_language, task_category = lang_category_extraction(path_to_file_for_probing)

        self.log_info = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
        self.log_info['params']['probing_task'] = probe_task
        self.log_info['params']['file_path'] = path_to_file_for_probing
        self.log_info['params']['task_language'] = task_language
        self.log_info['params']['task_category'] = task_category
        self.log_info['params']['probing_type'] = self.probing_type
        self.log_info['params']['batch_size'] = self.batch_size
        self.log_info['params']['hf_model_name'] = self.hf_model_name
        self.log_info['params']['classifier_name'] = self.classifier_name
        self.log_info['params']['metric_names'] = self.metric_names
        self.log_info['params']['original_classes_ratio'] = get_ratio_by_classes(task_dataset)

        if verbose:
            print('=' * 100)
            print(f'Task in progress: {probe_task}\nPath to data: {path_to_file_for_probing}')

        start_time = time()
        encode_func =  lambda x: self.transformer_model.encode_text(x, self.embedding_type)
        train = EncodeLoader(task_dataset["tr"], encode_func, self.batch_size, shuffle = self.shuffle)
        val = EncodeLoader(task_dataset["va"], encode_func, self.batch_size, shuffle = self.shuffle)
        test = EncodeLoader(task_dataset["te"], encode_func, self.batch_size, shuffle = self.shuffle)
        self.log_info['params']['encoded_labels'] = train.encoded_labels

        probing_iter_range = trange(num_layers, desc="Probing by layers") if verbose else range(num_layers)
        self.log_info['results']['elapsed_time(sec)'] = 0
        for layer in probing_iter_range:
            self.classifier = self.get_classifier(self.classifier_name, num_classes, self.transformer_model.config.hidden_size)
            self.criterion = torch.nn.CrossEntropyLoss()
            self.optimizer = AdamW(self.classifier.parameters())

            for epoch in range(train_epochs):
                epoch_train_loss = self.train(train.dataset, layer)
                epoch_val_loss, epoch_val_score = self.evaluate(val.dataset, layer, save_checkpoints)

                self.log_info['results']['train_loss'][layer].append(epoch_train_loss)
                self.log_info['results']['val_loss'][layer].append(epoch_val_loss)

                for m in self.metric_names:
                    self.log_info['results']['val_score'][m][layer].append(epoch_val_score[m])

            _, epoch_test_score = self.evaluate(test.dataset, layer, save_checkpoints)

            for m in self.metric_names:
                self.log_info['results']['test_score'][m][layer].append(epoch_test_score[m])
            self.log_info['results']['test_score'][layer].append(epoch_test_score)
        
        self.log_info['results']['elapsed_time(sec)'] = time() - start_time
        output_path = save_log(self.log_info, probe_task)
        if verbose:
            print(f"Experiments were saved in the folder: {output_path}")
