from probing.pipeline import ProbingPipeline
import glob
from tqdm import tqdm
from pathlib import Path
import logging

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch

experiment = ProbingPipeline(
    metric_names = ["f1", "accuracy"],
    encode_batch_size = 8
)

def get_max_memory_per_gpu_dict(dtype, model_name):
    """ try to generate the memory map based on what we know about the model and the available hardware """

    # figure out the memory map - the minimum per gpu required to load the model
    n_gpus = torch.cuda.device_count()

    if model_name == "bigscience/bloom" and n_gpus == 8 and torch.cuda.get_device_properties(0).total_memory > 79*2**30:
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



model_name = "bigscience/bloom"
dtype = torch.bfloat16

experiment.transformer_model.config = AutoConfig.from_pretrained(
            model_name, output_hidden_states=True, 
            output_attentions=True
            )

experiment.transformer_model.model = AutoModelForCausalLM.from_pretrained(
    model_name,
    config = experiment.transformer_model.config,
    device_map="auto",
    torch_dtype=dtype,
    max_memory = get_max_memory_per_gpu_dict(dtype, model_name)
)

experiment.transformer_model.tokenizer = AutoTokenizer.from_pretrained(
    model_name, config = experiment.transformer_model.config
)

experiment.transformer_model.device = "cuda:0"

bloom_langs = ['Chi Tumbuka',
 'Kikuyu',
 'Bambara',
 'Akan',
 'Xitsonga',
 'Sesotho',
 'Chi Chewa',
 'Twi',
 'Setswana',
 'Lingala',
 'Northern Sotho',
 'Fon',
 'Kirundi',
 'Wolof',
 'Luganda',
 'Chi Shona',
 'Isi Zulu',
 'Igbo',
 'Xhosa',
 'Kinyarwanda',
 'Yoruba',
 'Swahili',
 'Assamese',
 'Odia',
 'Gujarati',
 'Marathi',
 'Punjabi',
 'Kannada',
 'Nepali',
 'Telugu',
 'Malayalam',
 'Urdu',
 'Tamil',
 'Bengali',
 'Hindi',
 'Basque',
 'Indonesian',
 'Catalan',
 'Vietnamese',
 'Arabic',
 'Portuguese',
 'Spanish',
 'Code',
 'French',
 'Chinese',
 'English']


langs = bloom_langs
for lang in langs:
    tasks_files = glob.glob(f'/home/jovyan/UD*/*{lang}*/*.csv', recursive=True)
    print(lang, len(tasks_files))

    for f in tqdm(tasks_files):
        try:
            experiment.run(probe_task = Path(f).stem, path_to_task_file = f, verbose=True, train_epochs=20, is_scheduler = True)
        except Exception as e:
            logging.exception(e)