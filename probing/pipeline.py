from enum import Enum
from typing import List, Optional, Tuple
import os
from tqdm import trange
import numpy as np
import json
import torch
import pathlib
from datetime import datetime
from torch.utils.data import DataLoader

from classifier import LogReg, MLP
from data_former import DataFormer, EncodeLoader
from encoder import TransformersLoader
from metric import Metric
import config


class ProbingPipeline:
    def __init__(
        self,
        probing_type: Enum,
        hf_model_name: Enum,
        device: Optional[Enum] = None,
        classifier_name: Enum = "mlp",
        dropout_rate: float = 0.2,
        num_hidden: int = 256,
        batch_size: Optional[int] = 128,
        shuffle: bool = False,
        metric_name: Enum = "accuracy"
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

        self.metric = Metric(metric_name).metric
        self.transformer_model = TransformersLoader(hf_model_name, device)
        if device is None:
            self.device = self.transformer_model.device

        self.log_info = {}
    
    def get_classifier(self, classifier_name: Enum, num_classes: int):
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
            return self.get_classifier("mlp")

    def train(self, train_loader, layer):
        epoch_train_losses = []
        self.classifier.train()
        for x, y in train_loader:
            x = torch.squeeze(x[layer], 0).to(self.device)
            y = torch.tensor(y).to(self.device)

            prediction = self.classifier(x)
            loss = self.criterion(prediction, y)
            epoch_train_losses.append(loss.item())
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return np.mean(epoch_train_losses)
    
    def evaluate(
        self,
        dataloader: DataLoader,
        layer: int,
        save_checkpoints: bool
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

                epoch_predictions += prediction.data.max(1).indices.cpu()
                epoch_true_labels += y.cpu()
        
        metric_score = self.metric(epoch_predictions, epoch_true_labels).item()
        return np.mean(epoch_losses), metric_score

    def run(
        self,
        probe_task: Enum,
        path_to_task_file: Optional[os.PathLike] = None,
        train_epochs: int = 10,
        save_checkpoints: bool = False
    ):
        print(f'Task in progress: {probe_task}')
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

        self.log_info[probe_task]['params']['file_path'] = task_data.data_path

        encode_func =  self.transformer_model.encode_text
        train = EncodeLoader(task_dataset["tr"], encode_func, self.batch_size)
        val = EncodeLoader(task_dataset["va"], encode_func, self.batch_size)
        test = EncodeLoader(task_dataset["te"], encode_func, self.batch_size)
        train_loader = train.dataset
        val_loader = val.dataset
        test_loader = test.dataset
        self.encoded_labels = train.encoded_labels

        for layer in range(num_layers):
            self.log_info[probe_task]['results']['train_loss'][layer] = []
            self.log_info[probe_task]['results']['val_loss'][layer] = []
            self.log_info[probe_task]['results']['val_score'][layer] = []
            self.log_info[probe_task]['results']['test_score'][layer] = []

            self.classifier = self.get_classifier(self.classifier_name, num_classes)
            self.criterion = torch.nn.CrossEntropyLoss()
            self.optimizer = torch.optim.Adam(self.classifier.parameters())

            for epoch in trange(train_epochs):
                epoch_train_loss = self.train(train_loader, layer)
                epoch_val_loss, epoch_val_score = self.evaluate(val_loader, layer, save_checkpoints)

                self.log_info[probe_task]['results']['train_loss'][layer].append(epoch_train_loss)
                self.log_info[probe_task]['results']['val_loss'][layer].append(epoch_val_loss)
                self.log_info[probe_task]['results']['val_score'][layer].append(epoch_val_score)

            _, epoch_test_score = self.evaluate(test_loader, layer, save_checkpoints)
            self.log_info[probe_task]['results']['test_score'][layer].append(epoch_test_score)

        date = datetime.now().strftime("%Y_%m_%d-%I:%M:%S_%p")
        experiments_path = pathlib.Path(config.results_folder, f'{probe_task}_{date}')
        os.makedirs(experiments_path)
        
        log_path = pathlib.Path(experiments_path, "log.json")
        with open(log_path, "w") as outfile:
            json.dump(self.log_info, outfile, indent = 4)
        print('Experiments were saved in folder: ', str(experiments_path))
