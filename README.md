# Universal Probing Framework

This probing Framework provides a full pipeline for probing experiments, i. e. experiments for interpretation of large language models. In a nutshell, Probing Framework supports:

- automatic generation of probing tasks on the basis of [Universal Dependencies](https://universaldependencies.org/) annotation;
- generation of probing tasks based on manual queries to data in the [CONLL-U](https://universaldependencies.org/format.html) format;
- basic probing experiments with several classifers, such as Logistic Regression and Multilayer Perceptron;
- other probing methods, such as Minimum Description Length (MDL);
- baselines for probing experiments, such as label shuffling;
- different metrics, including standard ones (such as F1-score and accuracy) and selectivity (the difference between experiments and control tasks);
- visualisation and aggregation tools for further analysis of experiments.


### Getting started

1. Clone the repository with code:

```python
git clone https://github.com/AIRI-Institute/Probing_framework
cd Probing_framework/ 
```

2. Install requirements and appropriate torch version:

```python
bash cuda_install_requirements.sh
```

3. Install all other necessary packages:

```python
pip install -r requirements.txt
```
4. Check out this very comprehensive colab example: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1qJzLjWN8oWCsaTGKMoGNGSHFS6au6KMd#scrollTo=3_r8gilG2y3Y&uniqifier=1)

### More details and examples

| Section | Description |
|-|-|
| [About probing](docs/about_probing.md) | General information about probing|
| [About framework](docs/about_framework.md) | General information about this framework|
| [Web interface](docs/web.md) | Information about visualization part|
| [How to use](docs/usage.md) | Information with usage examples|


### How to cite

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
