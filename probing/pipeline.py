from enum import Enum
from typing import List, Optional, Tuple
import os
from tqdm import trange
import numpy as np
import json
import torch
import pathlib
from datetime import datetime

from probing.classifier import LogReg, MLP
from probing.data_former import DataFormer, EncodeLoader
from probing.encoder import TransformersLoader
from probing.metric import Metric
from probing import config


class ProbingPipeline:
    def __init__(
        self,
        probing_type: Enum,
        hf_model_name: Enum,
        device: Optional[Enum] = None,
        classifier_name: Enum = "logreg",
        dropout_rate: float = 0.2,
        num_hidden: int = 256,
        batch_size: Optional[int] = 128,
        shuffle: bool = False,
        metric_name: Enum = 'accuracy'
    ):
        self.hf_model_name = hf_model_name
        self.probing_type = probing_type
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.dropout_rate = dropout_rate
        self.num_hidden = num_hidden
        self.classifier_name = classifier_name
        self.metric = Metric(metric_name).metric

        self.transformer_model = TransformersLoader(hf_model_name, device)


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
            y = y.to(self.device)

            prediction = self.clf(x)
            loss = self.criterion(prediction, y)
            epoch_train_losses.append(loss.item())
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return np.mean(epoch_train_losses)
    
    def evaluate(self, dataloader, layer):
        epoch_losses = []
        epoch_predictions = []
        epoch_true_labels = []
        self.classifier.eval()
        with torch.no_grad():
            for x, y in dataloader:
                x = torch.squeeze(x[layer], 0).to(self.device)
                y = y.to(self.device)

                prediction = self.clf(x)
                loss = self.criterion(prediction, y)
                epoch_losses.append(loss.item())

                epoch_predictions += prediction.cpu()
                epoch_true_labels += y.cpu()
        
        metric_score = self.metric(epoch_predictions, epoch_true_labels).item()
        return np.mean(epoch_losses), metric_score

    def run(
        self,
        probe_task: Enum,
        path_to_task_file: Optional[os.PathLike] = None,
        train_epochs: int = 10
    ):
        print(f'Task in progress: {probe_task}')
        num_layers = self.transformer_model.config.num_hidden_layers
        self.log_info[probe_task] = {}
        self.log_info[probe_task]['file_path'] = path_to_task_file
        self.log_info[probe_task]['train_loss'] = {}
        self.log_info[probe_task]['val_loss'] = {}
        self.log_info[probe_task]['val_score'] = {}
        self.log_info[probe_task]['test_score'] = {}

        task_data = DataFormer(probe_task, path_to_task_file)
        task_dataset = task_data.samples
        num_classes = task_data.num_classes

        self.classifier = self.get_classifier(self.classifier_name, num_classes)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.classifier.parameters())

        encode_func =  self.transformer_model.encode_text
        train = EncodeLoader(task_dataset["tr"], encode_func, self.batch_size)
        val = EncodeLoader(task_dataset["va"], encode_func, self.batch_size)
        test = EncodeLoader(task_dataset["te"], encode_func, self.batch_size)
        train_loader = train.dataset
        val_loader = val.dataset
        test_loader = test.dataset

        for layer in range(num_layers):
            self.log_info[probe_task]['train_loss'][layer] = []
            self.log_info[probe_task]['val_loss'][layer] = []
            self.log_info[probe_task]['val_score'][layer] = []
            self.log_info[probe_task]['test_score'][layer] = []

            for epoch in trange(train_epochs):
                epoch_train_loss = self.train(train_loader, layer)
                epoch_val_loss, epoch_val_score = self.evaluate(val_loader, layer)

                self.log_info[probe_task]['train_loss'][layer].append(epoch_train_loss)
                self.log_info[probe_task]['val_loss'][layer].append(epoch_val_loss)
                self.log_info[probe_task]['val_score'][layer].append(epoch_val_score)

            _, epoch_test_score = self.evaluate(test_loader, layer)
            self.log_info[probe_task]['test_score'][layer].append(epoch_test_score)

        os.makedirs(config.results_folder)
        date = datetime.now().strftime("%Y_%m_%d-%I:%M:%S_%p")
        log_path = pathlib.Path(config.results_folder, f'{probe_task}_{date}.json')
        with open(log_path, "w") as outfile:
            json.dump(self.log_info, outfile, indent = 4)
        print('Experiments were saved as ', str(log_path))
