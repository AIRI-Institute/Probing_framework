import glob
from tqdm import tqdm
from pathlib import Path
import random
import numpy as np
import traceback
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch
import fire
import uuid
from typing import Optional
import gc
import os

from probing.pipeline import ProbingPipeline


def init_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def get_max_memory_per_gpu_dict(dtype, model_name):
    """ try to generate the memory map based on what we know about the model and the available hardware """

    # figure out the memory map - the minimum per gpu required to load the model
    n_gpus = torch.cuda.device_count()

    if n_gpus >= 8 and torch.cuda.get_device_properties(0).total_memory > 79*2**30:
        # hand crafted optimized memory map for 8x80 setup over BLOOM
        # this works with bs=40
        return {0: '0GIB', 1: '51GIB', 2: '51GIB', 3: '51GIB', 4: '51GIB', 5: '51GIB', 6: '51GIB', 7: '51GIB'}

    try:
        # model_params calculation, as we don't have a model yet to do:
        #model_params = sum(dict((p.data_ptr(), p.numel()) for p in model.parameters()).values())

        config = AutoConfig.from_pretrained(model_name)
        h = config.n_embed
        l = config.n_layer
        v = config.vocab_size
        # from https://github.com/bigscience-workshop/bigscience/tree/6917a3b5fefcf439d3485ca184b4d9f6ab605150/math#model-sizing
        model_params = l*(12*h**2 + 13*h) + v*h + 4*h
    except:
        print(f"The model {model_name} has a broken config file. Please notify the owner")
        raise

    bytes = torch.finfo(dtype).bits / 8
    param_memory_total_in_bytes = model_params * bytes
    # add 5% since weight sizes aren't the same and some GPU may need more memory
    param_memory_per_gpu_in_bytes = int(param_memory_total_in_bytes / n_gpus * 1.05)
    print(f"Estimating {param_memory_per_gpu_in_bytes/2**30:0.2f}GB per gpu for weights")

    # check the real available memory
    # load cuda kernels first and only measure the real free memory after loading (shorter by ~2GB)
    torch.ones(1).cuda()
    max_memory_per_gpu_in_bytes = torch.cuda.mem_get_info(0)[0]
    if max_memory_per_gpu_in_bytes < param_memory_per_gpu_in_bytes:
        raise ValueError(f"Unable to generate the memory map automatically as the needed estimated memory per gpu ({param_memory_per_gpu_in_bytes/2**30:0.2f}GB) is bigger than the available per gpu memory ({max_memory_per_gpu_in_bytes/2**30:0.2f}GB)")

    return {i: param_memory_per_gpu_in_bytes for i in range(torch.cuda.device_count())}


def main(
    model_name: str = "bigscience/bloom",
    UD_folder: os.PathLike = Path("/home/jovyan/datasets/UD/"),
    encoding_batch_size: int = 16,
    classifier_batch_size: int = 64,
    classifier_device: Optional[str] = "cuda:0" # all calculations here
):
    experiment = ProbingPipeline(
        metric_names = ["f1", "accuracy"],
        encoding_batch_size = encoding_batch_size,
        classifier_batch_size = classifier_batch_size
    )
    # init
    init_seed()
    unique_id = uuid.uuid4().hex

    dtype = torch.bfloat16
    experiment.transformer_model.config = AutoConfig.from_pretrained(
        model_name,
        output_hidden_states=True, 
        output_attentions=True
        )
    experiment.transformer_model.model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=experiment.transformer_model.config,
        device_map="auto",
        torch_dtype=dtype,
        max_memory = get_max_memory_per_gpu_dict(dtype, model_name)
    )
    experiment.transformer_model.tokenizer = AutoTokenizer.from_pretrained(
        model_name, config = experiment.transformer_model.config
    )
    experiment.transformer_model.device = classifier_device


    bloom_langs = [
        'Bambara', 'Wolof', 'Yoruba', 'Marathi',
        'Urdu', 'Tamil', 'Bengali', 'Hindi', 'Basque',
        'Indonesian', 'Catalan', 'Arabic', 'Portuguese',
        'Spanish', 'French', 
        'Chinese', 'English'
    ]
    for lang in tqdm(bloom_langs, desc="Processing by languages"):
        experiment.transformer_model.cache.clear() # clear cache for optimal work before a new language
        torch.cuda.empty_cache()
        gc.collect()

        tasks_files = glob.glob(f'{UD_folder}UD*/*{lang}*/*.csv', recursive=True)
        for f in tqdm(tasks_files):
            try:
                experiment.run(probe_task = Path(f).stem, path_to_task_file = f, verbose=True, train_epochs=20)
            except Exception:
                e = traceback.format_exc()

                curr_path = Path().absolute()
                log_errors = open(Path(curr_path, f"logErrors_{unique_id}.txt"), "a")
                log_errors.write(f + "\n")
                log_errors.write(e + "\n")
                log_errors.close()


if __name__ == "__main__":
    fire.Fire(main)