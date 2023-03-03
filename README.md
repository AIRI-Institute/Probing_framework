# Probing-framework
This framework includes:
* [UD-parser](https://github.com/AIRI-Institute/Probing_framework/blob/main/probing/ud_parser/README.md): Converter of conllu files from Universal Dependencies to the appropriate format for probing.
* [UD-filter]
* [Probing framework itself]

### Install requirements and appropriate torch version 
```
bash cuda_install_requirements.sh
```

### Usage examples of Probing framework:
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

# Our articles:
 - Original paper of this framework: https://aclanthology.org/2022.blackboxnlp-1.37/
    ```
    @inproceedings{serikov-etal-2022-universal,
        title = "Universal and Independent: Multilingual Probing Framework for Exhaustive Model Interpretation and Evaluation",
        author = "Serikov, Oleg  and
        Protasov, Vitaly  and
        Voloshina, Ekaterina  and
        Knyazkova, Viktoria  and
        Shavrina, Tatiana",
        booktitle = "Proceedings of the Fifth BlackboxNLP Workshop on Analyzing and Interpreting Neural Networks for NLP",
        month = dec,
        year = "2022",
        address = "Abu Dhabi, United Arab Emirates (Hybrid)",
        publisher = "Association for Computational Linguistics",
        url = "https://aclanthology.org/2022.blackboxnlp-1.37",
        pages = "441--456",
        abstract = "Linguistic analysis of language models is one of the ways to explain and describe their reasoning, weaknesses, and limitations. In the probing part of the model interpretability research, studies concern individual languages as well as individual linguistic structures. The question arises: are the detected regularities linguistically coherent, or on the contrary, do they dissonate at the typological scale? Moreover, the majority of studies address the inherent set of languages and linguistic structures, leaving the actual typological diversity knowledge out of scope.In this paper, we present and apply the GUI-assisted framework allowing us to easily probe massive amounts of languages for all the morphosyntactic features present in the Universal Dependencies data. We show that reflecting the anglo-centric trend in NLP over the past years, most of the regularities revealed in the mBERT model are typical for the western-European languages. Our framework can be integrated with the existing probing toolboxes, model cards, and leaderboards, allowing practitioners to use and share their familiar probing methods to interpret multilingual models.Thus we propose a toolkit to systematize the multilingual flaws in multilingual models, providing a reproducible experimental setup for 104 languages and 80 morphosyntactic features.",
    }
    ```