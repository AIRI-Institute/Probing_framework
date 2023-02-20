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
    splitter.convert(path_dir_conllu=<folder path>)

    # Or you can pass paths to each of three possible conllu files
    splitter.convert(tr_path=..., va_path=..., te_path=...)
    ```

* __Output__:
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


### Usage examples:
Check out [```probing/scripts```](https://github.com/AIRI-Institute/Probing_framework/tree/main/scripts) for the samples how to launch
* __Jupyter__:
    ```python3
    from probing.pipeline import ProbingPipeline

    experiment = ProbingPipeline(
        hf_model_name="bert-base-uncased",
        device="cuda:1",
        classifier_name="logreg",
        )

    experiment.run(probe_task="sent_len")
    ```

* __Output__:
    ```
    Task in progress: sent_len.
    Path to data: /home/jovyan/test/TEST/Probing_framework/data/sent_len.txt
    Data encoding train: 100%|████████████████████████████████████████████████████████████████████████████████| 782/782 [01:26<00:00,  9.06it/s]
    Data encoding val: 100%|██████████████████████████████████████████████████████████████████████████████████| 79/79 [00:08<00:00,  8.94it/s]
    Data encoding test: 100%|██████████████████████████████████████████████████████████████████████████████████| 79/79 [00:08<00:00,  9.33it/s]
    Probing by layers: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 12/12 [04:04<00:00, 20.40s/it]
    Experiments were saved in folder:  /home/test/test/TEST/Probing_framework/results/sent_len_2022_02_18-05:51:47_PM
    ```
