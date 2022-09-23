import csv
import os
import numpy as np
from collections import Counter
from enum import Enum
from pathlib import Path
from typing import List, Any, Dict, Optional
from sklearn.model_selection import train_test_split


def filter_labels_after_split(labels: List[Any]) -> List[Any]:
    """Skipping those classes which have only 1 sentence???"""

    labels_repeat_dict = Counter(labels)
    n_repeat = 1  # threshold to overcome further splitting problem
    return [label for label, count in labels_repeat_dict.items() if count > n_repeat]


def subsamples_split(
        probing_dict: Dict[str, List[str]],
        partition: List[float],
        random_seed: int,
        shuffle=True,
        split: List[Enum] = None
) -> Dict:
    """
    Splits data into three sets: train, validation, and test
    in the given relation
    Args:
        probing_dict: {class_label: [sentences]}
        partition: a relation that sentences should be split in. ex: [0.8, 0.1, 0.1]
        random_seed: random seed for splitting
        shuffle: if sentences should be randomly shuffled
        split: parts that data should be split to
    Returns:
        parts: {part: [[sentences], [labels]]}
    """
    num_classes = len(probing_dict.keys())
    probing_data = [(s, class_name) for class_name, sentences in probing_dict.items() if
                    len(sentences) > num_classes for s in sentences]
    if not probing_data:
        raise Exception('Some class has less sentences than the number of classes')
    parts = {}
    data, labels = map(np.array, zip(*probing_data))
    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, stratify=labels, train_size=partition[0],
        shuffle=shuffle, random_state=random_seed
    )

    if len(partition) == 2:
        parts = {
            split[0]: [X_train, y_train],
            split[1]: [X_test, y_test]
        }
    else:
        filtered_labels = filter_labels_after_split(y_test)
        if len(filtered_labels) >= 2:
            X_train = X_train[np.isin(y_train, filtered_labels)]
            y_train = y_train[np.isin(y_train, filtered_labels)]
            X_test = X_test[np.isin(y_test, filtered_labels)]
            y_test = y_test[np.isin(y_test, filtered_labels)]

            val_size = partition[1] / (1 - partition[0])
            if y_test.size != 0:
                X_val, X_test, y_val, y_test = train_test_split(
                    X_test, y_test, stratify=y_test, train_size=val_size,
                    shuffle=shuffle, random_state=random_seed
                )
                parts = {
                    split[0]: [X_train, y_train],
                    split[1]: [X_test, y_test],
                    split[2]: [X_val, y_val]
                }
    return parts


def read(path):
    with open(path, encoding='utf-8') as f:
        conllu_file = f.read()
    return conllu_file


def writer(partition_sets: Dict, task_name: str, language: str, save_path_dir: os.PathLike) -> Path:
    """
    Writes to a file
    Args:

        partition_sets: {part: [[sentences], [labels]]}
        task_name: name for the probing task (will be used in result file name)

    """
    result_path = Path(Path(save_path_dir).resolve(), f'{language}_{task_name}.csv')
    with open(result_path, 'w', encoding='utf-8') as newf:
        my_writer = csv.writer(newf, delimiter='\t', lineterminator='\n')
        for part in partition_sets:
            for sentence, value in zip(*partition_sets[part]):
                my_writer.writerow([part, value, sentence])
    return result_path


def extract_lang_from_udfile_path(
        ud_file_path: os.PathLike,
        language: Optional[str]
) -> str:
    """Extracts language from conllu file name"""

    if not language:
        return Path(ud_file_path).stem.split('-')[0]
    return language


def determine_ud_savepath(
        path_from_files: os.PathLike,
        save_path_dir: Optional[os.PathLike]
) -> Path:
    """Creates a path to save the result file (the same directory where conllu paths are stored"""

    final_path = None
    if not save_path_dir:
        final_path = path_from_files
    else:
        final_path = save_path_dir
    os.makedirs(final_path, exist_ok=True)
    return Path(final_path)


def delete_duplicates(probing_dict):
    all_sent = [s for cl_sent in probing_dict.values() for s in cl_sent]
    duplicates = [item for item, count in Counter(all_sent).items() if count > 1]
    new_probing_dict = {}
    for cl in probing_dict:
        new_probing_dict[cl] = [s for s in probing_dict[cl] if s not in duplicates]
    return new_probing_dict
