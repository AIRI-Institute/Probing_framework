import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import fire
import numpy as np

from probing.ud_filter.filtering_probing import ProbingConlluFilter

# adverbial clausal modifiers VS clausal modifiers of a noun
complex_query: Dict[str, Tuple[dict, dict]] = {
    "ADVCL": ({"H": {}, "CL": {}}, {("H", "CL"): {"deprels": "^advcl$"}}),
    "ACL": (
        {
            "H": {},
            "CL": {},
        },
        {("H", "CL"): {"deprels": "^acl$"}},
    ),
}


def init_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)


def main(
    path_to_conllu_files: str = "conllu_files_dir",
    queries: Dict[str, Tuple[dict, dict]] = complex_query,
    save_dir_path: Optional[os.PathLike[Any]] = Path(
        Path(__file__).parent.resolve(), "advcl_acl_prob_tasks"
    ),
    task_name: str = "advcl_acl",
    partition: List[float] = [0.8, 0.1, 0.1],
):
    probing_filter = ProbingConlluFilter()
    probing_filter.upload_files(dir_conllu_path=path_to_conllu_files)
    probing_filter.filter_and_convert(
        queries=queries,
        task_name=task_name,
        save_dir_path=save_dir_path,
        partition=partition,
    )


if __name__ == "__main__":
    fire.Fire(main)
