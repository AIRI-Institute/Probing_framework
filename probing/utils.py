from enum import Enum
from typing import Optional, Dict, Any
import os
import glob
import pathlib
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
        raise RuntimeError(f"Provided path: {file_path} doesn\'t exist.")
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
    experiments_path = pathlib.Path(config.results_folder, f'{probe_task}_{date}')
    os.makedirs(experiments_path)
    
    log_path = pathlib.Path(experiments_path, "log.json")
    with open(log_path, "w") as outfile:
        json.dump(log, outfile, indent = 4, default = myconverter)
    print("Experiments were saved in the folder: ", str(experiments_path))
