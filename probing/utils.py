import gc
import glob
import json
import os
import pathlib
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from probing import config


def clear_memory():
    torch.cuda.empty_cache()
    gc.collect()


def get_probe_task_path(
    probe_task_name: str, file_path: Optional[os.PathLike] = None
) -> os.PathLike:
    if file_path is None:
        path_to_folder = pathlib.Path(config.DATA_FOLDER_PATH, probe_task_name)
        path_to_file = glob.glob(f"{path_to_folder}*")

        if len(path_to_file) == 0:
            raise RuntimeError(
                f"We didn't find any files for the task: {probe_task_name}."
                "You should provide a path to the file with data."
            )
        return pathlib.Path(path_to_file[0])

    if not os.path.exists(file_path):
        raise RuntimeError(f"Provided path: {file_path} doesn't exist")
    return file_path


def lang_category_extraction(
    file_path: os.PathLike,
) -> Tuple[Optional[str], Optional[str]]:
    if "_" in str(file_path):
        path = str(pathlib.Path(file_path).stem)
        task_language = path.split("_")[0]
        task_category = path.split("_")[-1]
        return task_language, task_category
    return None, None


class ProbingLog(defaultdict):
    def __init__(self, *args, **kwargs):
        super(ProbingLog, self).__init__(ProbingLog, *args, **kwargs)
        self.start_time = ProbingLog.get_time()
        self.results_folder = pathlib.Path(
            config.HOME_PATH, f"probing_results/experiment_{self.start_time}"
        )

    def __repr__(self):
        return repr(dict(self))

    def add(self, key, value):
        if key not in self:
            self[key] = []
        self[key].append(value)

    @staticmethod
    def get_time() -> str:
        return datetime.now().strftime("%Y_%m_%d-%I:%M:%S_%p")

    @staticmethod
    def myconverter(obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, datetime):
            return obj.__str__()
        if isinstance(obj, pathlib.PosixPath):
            return obj.__str__()
        return obj

    def save_log(self, probe_task: str) -> os.PathLike:
        saving_date = ProbingLog.get_time()
        experiments_path = pathlib.Path(
            self.results_folder, f"{saving_date}_{probe_task}"
        )
        if not probe_task.startswith("test_"):
            os.makedirs(experiments_path, exist_ok=True)
            log_path = pathlib.Path(experiments_path, "log.json")

            with open(log_path, "w") as outfile:
                json.dump(self, outfile, indent=4, default=ProbingLog.myconverter)
        return experiments_path


def kl_divergence(z, mu_theta, p_theta):
    log_prior = torch.distributions.Normal(0, 1).log_prob(z)
    log_p_q = torch.distributions.Normal(
        mu_theta, torch.log(1 + torch.exp(p_theta))
    ).log_prob(z)
    return (log_p_q - log_prior).mean()


class KL:
    accumulated_kl_div = 0


class KL_Loss:
    def __init__(self, blank_token: int = 0, loss=None):
        self.blank = blank_token
        self.loss = loss

    def __call__(self, y_true, y_pred, model=None, **kwargs):
        kl = model.accumulated_kl_div
        model.reset_kl_div()
        return self.loss + kl
