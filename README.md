# Probing-framework
Framework for probing tasks.

### Firstly
```
pip install -r requirements.txt
```

### Example of how it works:
* __Command Line__:
    ```
    python probing/main.py --probe_task "conj_type" --hf_model_name "bert-base-multilingual-cased"
    ```

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

experiment.run(probe_task = "sent_len", train_epochs = 10)
```

* __OUTPUT__:
    ```
    Task in progress: conj_type
    Data encoding: 100%|████████████████████████████████████████████████████████████████████████████████| 782/782 [01:26<00:00,  9.06it/s]
    Data encoding: 100%|██████████████████████████████████████████████████████████████████████████████████| 79/79 [00:08<00:00,  8.94it/s]
    Data encoding: 100%|██████████████████████████████████████████████████████████████████████████████████| 79/79 [00:08<00:00,  9.33it/s]
    Probing by layers... 100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 12/12 [04:04<00:00, 20.40s/it]
    Experiments were saved in folder:  /home/jovyan/protasov/AIRI/Probing_framework/results/conj_type_2022_02_18-05:51:47_PM
    ```