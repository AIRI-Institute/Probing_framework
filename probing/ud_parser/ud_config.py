from typing import List, Dict
from enum import Enum


too_much_files_err_str = "Too much files. You provided \"{}\" files"


partitions_by_files: Dict[int, List[List[float]]] = {
    1: [[0.8, 0.1, 0.1]],
    2: [[1.0], [0.5, 0.5]],
    3: [[1.0], [1.0], [1.0]]
    }


splits_by_files: Dict[int, List[List[Enum]]] = {
    1: [["tr", "va", "te"]],
    2: [["tr"], ["va", "te"]],
    3: [["tr"], ["va"], ["te"]]
    }
