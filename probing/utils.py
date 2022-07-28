from enum import Enum
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
from collections import Counter
import os
import glob
import logging
import pathlib
import torch
import numpy as np
import json
from datetime import datetime

from probing import config


def get_probe_task_path(
    probe_task_name: Enum,
    file_path: Optional[os.PathLike] = None
) -> os.PathLike:
    if file_path is None:
        path_to_folder = pathlib.Path(config.data_folder, probe_task_name)
        path_to_file = glob.glob(f'{path_to_folder}*')

        if len(path_to_file) == 0:
            raise RuntimeError(
                f"We didn\'t find any files for the task: {probe_task_name}."
                "You should provide a path to the file with data."
                )
        return path_to_file[0]

    elif not os.path.exists(file_path):
        raise RuntimeError(f"Provided path: {file_path} doesn\'t exist")
    return file_path


def myconverter(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, datetime.datetime):
        return obj.__str__()
    return obj


def save_log(log: Dict, probe_task: str) -> None:
    date = datetime.now().strftime("%Y_%m_%d-%I:%M:%S_%p")
    experiments_path = pathlib.Path(config.results_folder, f'{date}_{probe_task}')
    os.makedirs(experiments_path, exist_ok=True)
    
    log_path = pathlib.Path(experiments_path, "log.json")
    with open(log_path, "w") as outfile:
        json.dump(log, outfile, indent = 4, default = myconverter)
    return str(experiments_path)


def get_ratio_by_classes(samples: Dict[Enum, List[str]]) -> Dict[Enum, Dict[Enum, int]]:
    ratio_by_classes = {}
    for class_name in samples:
        class_labels_all = [i[1] for i in samples[class_name]]
        ratio_by_classes[class_name] = dict(Counter(class_labels_all))
    return ratio_by_classes

def lang_category_extraction(file_path: os.PathLike) -> Tuple[Optional[str], Optional[str]]:
    if '_' in file_path:   
        path = str(Path(file_path).stem)           
        task_language = path.split('_')[0]
        task_category = path.split('_')[-1]
    else:
        task_language, task_category = None, None
    return task_language, task_category


def exclude_rows(tensor: torch.Tensor, rows_to_exclude: List[int]) -> torch.Tensor:
    if len(tensor.size()) == 1:
        tensor = tensor.view(-1,1)

    tensor_shape = tensor.size()
    assert len(tensor_shape) == 2
    tensor = tensor.view(*tensor_shape,1)

    mask = torch.ones(tensor_shape, dtype=torch.bool)
    mask[rows_to_exclude, :] = False
    new_num_rows = tensor_shape[0] - len(rows_to_exclude)
    if new_num_rows == 0:
        logging.warning(f"All samples were excluded due to long sentences truncation")
        return tensor[mask]
    output = tensor[mask].view(new_num_rows, -1)
    return output
