from enum import Enum
from typing import Optional
import os
import glob
import pathlib

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
