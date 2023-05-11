import glob
from tqdm import tqdm
from pathlib import Path
import random
import numpy as np
import traceback
import torch
import fire
import uuid
import os
import gc
from typing import Optional

from probing.pipeline import ProbingPipeline


def init_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def main(
    model_name: str = "bert-base-multilingual-cased",
    UD_folder: os.PathLike = "/home/jovyan/datasets/UD/",
    encoding_batch_size: int = 64,
    classifier_batch_size: int = 64,
    device: Optional[str] = None,
):
    experiment = ProbingPipeline(
        hf_model_name=model_name,
        device=device,
        metric_names=["f1", "accuracy"],
        encoding_batch_size=encoding_batch_size,
        classifier_batch_size=classifier_batch_size,
    )
    # init
    init_seed()
    unique_id = uuid.uuid4().hex

    bloom_langs = [
        "Bambara",
        "Wolof",
        "Yoruba",
        "Marathi",
        "Urdu",
        "Tamil",
        "Bengali",
        "Hindi",
        "Basque",
        "Indonesian",
        "Catalan",
        "Arabic",
        "Portuguese",
        "Spanish",
        "French",
        "Chinese",
        "English",
    ]
    for lang in tqdm(bloom_langs, desc="Processing by languages"):
        experiment.transformer_model.Caching.cache.clear()  # clear cache for optimal work before a new language
        torch.cuda.empty_cache()
        gc.collect()

        tasks_files = glob.glob(f"{UD_folder}UD*/*{lang}*/*.csv", recursive=True)
        for f in tqdm(tasks_files):
            try:
                experiment.run(
                    probe_task=Path(f).stem,
                    path_to_task_file=f,
                    verbose=True,
                    train_epochs=20,
                )
            except Exception:
                e = traceback.format_exc()

                curr_path = Path().absolute()
                log_errors = open(Path(curr_path, f"logErrors_{unique_id}.txt"), "a")
                log_errors.write(f + "\n")
                log_errors.write(e + "\n")
                log_errors.close()


if __name__ == "__main__":
    fire.Fire(main)
