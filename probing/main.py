from enum import Enum
from typing import Optional, Union
import argparse
import os

from probing.pipeline import ProbingPipeline


def main(
    probing_type: Enum,
    hf_model_name: Enum,
    probe_task: Union[Enum, str],
    train_epochs: int = 10,
    save_checkpoints: bool = False,
    path_to_task_file: Optional[os.PathLike] = None,
    device: Optional[Enum] = None,
    classifier_name: Enum = "mlp",
    metric_name: Enum = "accuracy",
    embedding_type: Enum = "cls",
    batch_size: Optional[int] = 64,
    dropout_rate: float = 0.2,
    num_hidden: int = 250,
    shuffle: bool = True,
    truncation: bool = False
):
    experiment = ProbingPipeline(
        probing_type,
        hf_model_name,
        device,
        classifier_name,
        metric_name,
        embedding_type,
        batch_size,
        dropout_rate,
        num_hidden,
        shuffle,
        truncation
    )

    experiment.run(
        probe_task,
        path_to_task_file,
        train_epochs,
        save_checkpoints
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--probe_task',
        type=str,
        required=True,
        help="Name of the probing task"
    )

    parser.add_argument(
        '--path_to_task_file',
        type=str,
        default=None,
        help="Path to the file with data for experiments. You can choose both data fron \"data\" directory or pass yours",
    )

    parser.add_argument(
        '--probing_type',
        type=str,
        default="layer",
        help="Type of probing experiments."
    )

    parser.add_argument(
        '--hf_model_name',
        type=str,
        required=True,
        help="Name of the model from hugging-face. This models is used for embeddings extraction."
    )

    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help="Device: cuda:x /cpu."
    )

    parser.add_argument(
        '--classifier_name',
        type=str,
        default="logreg",
        help="Name of the classifier: mlpr or logreg."
    )

    parser.add_argument(
        '--metric_name',
        type=str,
        default="accuracy",
        help="Name of the metric for val and test parts."
    )

    parser.add_argument(
        '--embedding_type',
        type=str,
        default="cls",
        help="As the embedding of the sentence, we support different types of how to count it: avg, sum, cls."
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help="Batch size."
    )

    parser.add_argument(
        '--dropout_rate',
        type=float,
        default=0.2,
        help="Dropout rate."
    )

    parser.add_argument(
        '--num_hidden',
        type=int,
        default=250,
        help="As for MLP classifier, you can point out the dimension of the hidden layer."
    )

    parser.add_argument(
        '--train_epochs',
        type=int,
        default=10,
        help="Number of training epochs."
    )

    parser.add_argument(
        '--shuffle',
        type=bool,
        default=True,
        help="Whether or not to apply the shuffling for the data during train, val, test parts."
    )

    parser.add_argument(
        '--save_checkpoints',
        type=bool,
        default=False,
        help="Save model\'s checkpoints at each epoch or do not."
    )

    parser.add_argument(
        '--truncation',
        type=bool,
        default=False,
        help="Truncate or exclude long sentences."
    )

    args = parser.parse_args()
    main(
        args.probing_type,
        args.hf_model_name,
        args.probe_task,
        args.train_epochs,
        args.save_checkpoints,
        args.path_to_task_file,
        args.device,
        args.classifier_name,
        args.metric_name,
        args.embedding_type,
        args.batch_size,
        args.dropout_rate,
        args.num_hidden,
        args.shuffle
    )
