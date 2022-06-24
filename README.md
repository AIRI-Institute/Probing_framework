# Probing-framework
Framework for probing tasks.

### Firstly
```
pip install -r requirements.txt
```

### Example of how SentEval Converter works:
* __Jupyter__:
    ```python3
    from probing.ud_parser.ud_parser import ConlluUDParser

    splitter = ConlluUDParser()

    # You can provide a direct path to the folder with conllu files
    splitter.convert(dir_conllu_path=<folder path>)

    # Or you can pass paths to each of three possible conllu files
    splitter.convert(tr_path=..., va_path=..., te_path=...)
    ```





### Example of how Probing Engine works:
For more parameters you can check out ```probing/main.py```
* __Jupyter__:
    ```python3
    from probing.pipeline import ProbingPipeline

    experiment = ProbingPipeline(
        probing_type = "layer",
        hf_model_name = "bert-base-multilingual-cased",
        device = "cuda:0",
        classifier_name = "mlp",
        metric_name = "accuracy",
        embedding_type = "cls",
        batch_size = 256,
    )

    # In case of the custom data for task SentLen
    experiment.run(probe_task = "sent_len", train_epochs = 10, verbose=True)

    # In case you want to provide the folder with your probing files
    experiment.run(probe_task = <task name>, path_to_task_file = <path to file for probing task>, train_epochs = 10, verbose=True)
    ```

* __Command Line__:
    ```
    python probing/main.py --probe_task "conj_type" --hf_model_name "bert-base-multilingual-cased" --device "cuda:0"

    python probing/main.py --probe_task <task name> --hf_model_name "bert-base-multilingual-cased" --device "cuda:0" --path_to_task_file <path to file for probing task>
    ```


* __OUTPUT__:
    ```
    Task in progress: sent_len.
    Path to data: /home/jovyan/protasov/AIRI/Probing_framework/data/sent_len.txt
    Data encoding: 100%|████████████████████████████████████████████████████████████████████████████████| 782/782 [01:26<00:00,  9.06it/s]
    Data encoding: 100%|██████████████████████████████████████████████████████████████████████████████████| 79/79 [00:08<00:00,  8.94it/s]
    Data encoding: 100%|██████████████████████████████████████████████████████████████████████████████████| 79/79 [00:08<00:00,  9.33it/s]
    Probing by layers... 100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 12/12 [04:04<00:00, 20.40s/it]
    Experiments were saved in folder:  /home/jovyan/protasov/AIRI/Probing_framework/results/sent_len_2022_02_18-05:51:47_PM
    ```