from enum import Enum
from typing import Optional, Callable
import os
from tqdm import trange
import numpy as np
import torch
from torch.utils.data import DataLoader

from probing.classifier import LogReg, MLP
from probing.data_former import DataFormer, EncodeLoader
from probing.encoder import TransformersLoader
from probing.metric import Metric
from probing.utils import save_log


class ProbingPipeline:
    def __init__(
        self,
        probing_type: Enum,
        hf_model_name: Enum,
        device: Optional[Enum] = None,
        classifier_name: Enum = "mlp",
        metric_name: Enum = "accuracy",
        embedding_type: Enum = "cls",
        batch_size: Optional[int] = 128,
        dropout_rate: float = 0.2,
        num_hidden: int = 256,
        shuffle: bool = False
    ):
        self.hf_model_name = hf_model_name
        self.probing_type = probing_type
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.dropout_rate = dropout_rate
        self.num_hidden = num_hidden
        self.classifier_name = classifier_name
        self.metric_name = metric_name
        self.device = device
        self.embedding_type = embedding_type

        self.metric = Metric(metric_name).metric
        self.transformer_model = TransformersLoader(hf_model_name, device)
        if device is None:
            self.device = self.transformer_model.device

        self.log_info = {}

    def get_classifier(
        self,
        classifier_name: Enum,
        num_classes: int
    ) -> Callable:
        embed_dim = self.transformer_model.config.hidden_size
        if classifier_name == "logreg":
            return LogReg(
                input_dim = embed_dim,
                num_classes = num_classes
                ).to(self.device)
        elif classifier_name == "mlp":
            return MLP(
                input_dim = embed_dim,
                num_classes = num_classes,
                num_hidden =  self.num_hidden,
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
    ):
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

        epoch_metric_score = self.metric(epoch_predictions, epoch_true_labels).item()
        epoch_loss = np.mean(epoch_losses)
        return epoch_loss, epoch_metric_score

    def run(
        self,
        probe_task: Enum,
        path_to_task_file: Optional[os.PathLike] = None,
        train_epochs: int = 10,
        save_checkpoints: bool = False
    ) -> None:
        num_layers = self.transformer_model.config.num_hidden_layers
        self.log_info[probe_task] = {}
        self.log_info[probe_task]['results'] = {}
        self.log_info[probe_task]['params'] = {}
        self.log_info[probe_task]['results']['train_loss'] = {}
        self.log_info[probe_task]['results']['val_loss'] = {}
        self.log_info[probe_task]['results']['val_score'] = {}
        self.log_info[probe_task]['results']['test_score'] = {}

        self.log_info[probe_task]['params']['probing_type'] = self.probing_type
        self.log_info[probe_task]['params']['batch_size'] = self.batch_size
        self.log_info[probe_task]['params']['hf_model_name'] = self.hf_model_name
        self.log_info[probe_task]['params']['classifier_name'] = self.classifier_name
        self.log_info[probe_task]['params']['metric_name'] = self.metric_name

        task_data = DataFormer(probe_task, path_to_task_file)
        task_dataset = task_data.samples
        num_classes = task_data.num_classes

        print(f'Task in progress: {probe_task}.\nPath to data: {task_data.data_path}')
        self.log_info[probe_task]['params']['file_path'] = task_data.data_path

        encode_func =  lambda x: self.transformer_model.encode_text(x, self.embedding_type)
        train = EncodeLoader(task_dataset["tr"], encode_func, self.batch_size)
        val = EncodeLoader(task_dataset["va"], encode_func, self.batch_size)
        test = EncodeLoader(task_dataset["te"], encode_func, self.batch_size)
        train_loader = train.dataset
        val_loader = val.dataset
        test_loader = test.dataset
        self.log_info[probe_task]['results']['encoded_labels'] = train.encoded_labels

        for layer in trange(num_layers, desc="Probing by layers..."):
            self.log_info[probe_task]['results']['train_loss'][layer] = []
            self.log_info[probe_task]['results']['val_loss'][layer] = []
            self.log_info[probe_task]['results']['val_score'][layer] = []
            self.log_info[probe_task]['results']['test_score'][layer] = []

            self.classifier = self.get_classifier(self.classifier_name, num_classes)
            self.criterion = torch.nn.CrossEntropyLoss()
            self.optimizer = torch.optim.AdamW(self.classifier.parameters())

            for epoch in range(train_epochs):
                epoch_train_loss = self.train(train_loader, layer)
                epoch_val_loss, epoch_val_score = self.evaluate(val_loader, layer, save_checkpoints)

                self.log_info[probe_task]['results']['train_loss'][layer].append(epoch_train_loss)
                self.log_info[probe_task]['results']['val_loss'][layer].append(epoch_val_loss)
                self.log_info[probe_task]['results']['val_score'][layer].append(epoch_val_score)

            _, epoch_test_score = self.evaluate(test_loader, layer, save_checkpoints)
            self.log_info[probe_task]['results']['test_score'][layer].append(epoch_test_score)

        save_log(self.log_info)
