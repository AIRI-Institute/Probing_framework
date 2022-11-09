import glob
import json
import logging
import os
import pathlib
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from probing import config


def get_probe_task_path(
    probe_task_name: str, file_path: Optional[os.PathLike] = None
) -> os.PathLike:
    if file_path is None:
        path_to_folder = pathlib.Path(config.data_folder, probe_task_name)
        path_to_file = glob.glob(f"{path_to_folder}*")

        if len(path_to_file) == 0:
            raise RuntimeError(
                f"We didn't find any files for the task: {probe_task_name}."
                "You should provide a path to the file with data."
            )
        return pathlib.Path(path_to_file[0])

    elif not os.path.exists(file_path):
        raise RuntimeError(f"Provided path: {file_path} doesn't exist")
    return file_path


def myconverter(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, datetime):
        return obj.__str__()
    elif isinstance(obj, pathlib.PosixPath):
        return obj.__str__()
    return obj


def save_log(log: Dict, probe_task: str) -> os.PathLike:
    log_file_name = "log.json"
    date = datetime.now().strftime("%Y_%m_%d-%I:%M:%S_%p")
    experiments_path = pathlib.Path(config.results_folder, f"{date}_{probe_task}")
    if not probe_task.startswith("test_"):
        os.makedirs(experiments_path, exist_ok=True)
        log_path = pathlib.Path(experiments_path, log_file_name)

        with open(log_path, "w") as outfile:
            json.dump(log, outfile, indent=4, default=myconverter)
    return experiments_path


def lang_category_extraction(
    file_path: os.PathLike,
) -> Tuple[Optional[str], Optional[str]]:
    if "_" in str(file_path):
        path = str(pathlib.Path(file_path).stem)
        task_language = path.split("_")[0]
        task_category = path.split("_")[-1]
        return task_language, task_category
    return None, None


def exclude_rows(tensor: torch.Tensor, rows_to_exclude: torch.Tensor) -> torch.Tensor:
    if len(tensor.size()) == 1:
        tensor = tensor.view(-1, 1)

    tensor_shape = tensor.size()
    assert len(tensor_shape) == 2
    tensor = tensor.view(*tensor_shape, 1)

    mask = torch.ones(tensor_shape, dtype=torch.bool)
    mask[rows_to_exclude, :] = False
    new_num_rows = tensor_shape[0] - len(rows_to_exclude)
    if new_num_rows == 0:
        logging.warning(f"All samples were excluded due to long sentences truncation")
        return tensor[mask]
    output = tensor[mask].view(new_num_rows, -1)
    return output


class ProbingLog(defaultdict):
    def __init__(self, *args, **kwargs):
        super(ProbingLog, self).__init__(ProbingLog, *args, **kwargs)

    def __repr__(self):
        return repr(dict(self))

    def add(self, key, value):
        if key not in self:
            self[key] = []
        self[key].append(value)
        
def kl_divergence(z, mu_theta, p_theta):
    log_prior = torch.distributions.Normal(0, 1).log_prob(z) 
    log_p_q = torch.distributions.Normal(mu_theta, torch.log(1 + torch.exp(p_theta))).log_prob(z) 
    return (log_p_q - log_prior).mean()
 
class KL:
    accumulated_kl_div = 0


class KL_Loss:
    def __init__(self, blank_token: int = 0):
        self.blank = blank_token
        
    def __call__(self, y_true, y_pred, model = None, **kwargs):
        reconstruction_error = torch.nn.CrossEntropyLoss()(y_pred, y_true)
        kl = model.accumulated_kl_div 
        model.reset_kl_div()
        return reconstruction_error + kl

