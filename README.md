# Probing-framework
Framework for probing tasks.

### Install requirements and appropriate torch version 
```
bash cuda_install_requirements.sh
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

* __OUTPUT__:
    ```
    WARNING:root:Category "Abbr" has only one class
    WARNING:root:Category "AdpType" has only one class
    WARNING:root:The classes in train and validation parts are different for category "Case"
    WARNING:root:Category "Degree" has only one class
    WARNING:root:Category "Foreign" has only one class
    WARNING:root:Category "PartType" has only one class
    WARNING:root:Category "Poss" has only one class
    WARNING:root:The classes in train and test parts are different for category "PronType"
    WARNING:root:Category "Reflex" has only one class
    WARNING:root:The classes in train and validation parts are different for category "Tense"
    WARNING:root:Category "Variant" has only one class
    Writing to file: /home/jovyan/datasets/UD/UD/UD_Romanian-RRT/ro_rrt_Case.csv
    Writing to file: /home/jovyan/datasets/UD/UD/UD_Romanian-RRT/ro_rrt_Definite.csv
    Writing to file: /home/jovyan/datasets/UD/UD/UD_Romanian-RRT/ro_rrt_Gender.csv
    Writing to file: /home/jovyan/datasets/UD/UD/UD_Romanian-RRT/ro_rrt_Mood.csv
    Writing to file: /home/jovyan/datasets/UD/UD/UD_Romanian-RRT/ro_rrt_NumForm.csv
    Writing to file: /home/jovyan/datasets/UD/UD/UD_Romanian-RRT/ro_rrt_NumType.csv
    Writing to file: /home/jovyan/datasets/UD/UD/UD_Romanian-RRT/ro_rrt_Number.csv
    Writing to file: /home/jovyan/datasets/UD/UD/UD_Romanian-RRT/ro_rrt_Number[psor].csv
    Writing to file: /home/jovyan/datasets/UD/UD/UD_Romanian-RRT/ro_rrt_Person.csv
    Writing to file: /home/jovyan/datasets/UD/UD/UD_Romanian-RRT/ro_rrt_Polarity.csv
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

    python probing/main.py --probe_task <task name> --path_to_task_file <path to file for probing task> --hf_model_name "bert-base-multilingual-cased" --device "cuda:0"
    ```


* __OUTPUT__:
    ```
    Task in progress: sent_len.
    Path to data: /home/jovyan/test/TEST/Probing_framework/data/sent_len.txt
    Data encoding: 100%|████████████████████████████████████████████████████████████████████████████████| 782/782 [01:26<00:00,  9.06it/s]
    Data encoding: 100%|██████████████████████████████████████████████████████████████████████████████████| 79/79 [00:08<00:00,  8.94it/s]
    Data encoding: 100%|██████████████████████████████████████████████████████████████████████████████████| 79/79 [00:08<00:00,  9.33it/s]
    Probing by layers... 100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 12/12 [04:04<00:00, 20.40s/it]
    Experiments were saved in folder:  /home/test/test/TEST/Probing_framework/results/sent_len_2022_02_18-05:51:47_PM
    ```
