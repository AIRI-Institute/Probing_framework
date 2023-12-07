import csv
import os
import re
import typing
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from conllu import parse, parse_tree
from conllu.models import Token, TokenTree
from nltk.tokenize import wordpunct_tokenize
from sklearn.model_selection import train_test_split
from transformers.utils import logging

from probing.ud_parser.ud_config import (
    partitions_by_files,
    splits_by_files,
    too_much_files_err_str,
)

logging.set_verbosity_warning()
logger = logging.get_logger("probing_parser")


class ConlluUDParser:
    def __init__(
        self,
        shuffle: bool = True,
        verbose: bool = True,
        deprel: Optional[str] = None,
        upos: Optional[str] = None,
        sorting: Optional[str] = None,
    ):
        self.shuffle = shuffle
        self.verbose = verbose
        self.deprel = deprel
        self.upos = upos
        self.sorting = sorting

    def read(self, path: str) -> str:
        """
        Reads a file
        Args:
            path: a path to a file
        """
        with open(path, encoding="utf-8") as f:
            conllu_file = f.read()
        return conllu_file

    def writer(
        self,
        partition_sets: Dict,
        category: str,
        language: str,
        save_path_dir: os.PathLike,
    ) -> Path:
        """
        Writes to a file
        Args:
             result_path: a filename that will be generated
             partition_sets: the data split into 3 parts
        """
        result_path = Path(Path(save_path_dir).resolve(), f"{language}_{category}.csv")
        with open(result_path, "w", encoding="utf-8") as newf:
            my_writer = csv.writer(newf, delimiter="\t", lineterminator="\n")
            for part in partition_sets:
                for sentence_and_id, value in zip(*partition_sets[part]):
                    sentence, id = sentence_and_id
                    my_writer.writerow([part, value, sentence, id])
        return result_path

    def find_category_token(
        self, category: str, head: Token, children: List[TokenTree]
    ) -> Optional[Token]:
        """
        Finds a token that has a given category and is located on the top of a tree
        Args:
            category: a grammatical value that needed to be found
            head: the root of a sentence
            children: children of a token in token tree
        """
        if head["feats"] and category in head["feats"]:
            if (
                self.upos
                and head["upos"] == self.upos
                and self.deprel
                and head["deprel"] == self.deprel
            ):
                return head
            elif self.deprel and head["deprel"] == self.deprel:
                return head
            elif self.upos and head["upos"] == self.upos:
                return head
            elif not self.upos and not self.deprel:
                return head

        for token in children:
            token_info = token.token
            result = self.find_category_token(category, token_info, token.children)
            if result:
                return result
        return None

    def classify(
        self,
        token_trees: List[TokenTree],
        category: str,
        subcategory: str,
    ) -> Dict:
        """
        Classifies sentences by a grammatical value they contain
        Args:
            token_trees: sentences represented as trees
            category: a grammatical value
        """
        probing_data = defaultdict(list)
        for token_tree in token_trees:
            if token_tree.metadata:
                s_text = " ".join(wordpunct_tokenize(token_tree.metadata["text"]))
                root = token_tree.token
                category_token = self.find_category_token(
                    category, root, token_tree.children
                )
                if category_token:
                    if (
                        (self.sorting == None)
                        or (
                            self.sorting == "by_pos"
                            and category_token["upos"] == subcategory
                        )
                        or (
                            self.sorting == "by_deprel"
                            and category_token["deprel"] == subcategory
                        )
                    ):
                        value = category_token["feats"][category]
                        token_id = category_token["id"] - 1
                        probing_data[value].append((s_text, token_id))
                    elif self.sorting == "by_pos_and_deprel":
                        pos, deprel = subcategory.split("_")
                        if (
                            category_token["upos"] == pos
                            and category_token["deprel"] == deprel
                        ):
                            value = category_token["feats"][category]
                            token_id = category_token["id"] - 1
                            probing_data[value].append((s_text, token_id))
        return probing_data

    def filter_labels_after_split(self, labels: List[Any]) -> List[Any]:
        labels_repeat_dict = Counter(labels)
        n_repeat = 1  # threshold to overcome further splitting problem
        return [
            label for label, count in labels_repeat_dict.items() if count > n_repeat
        ]

    def check_parts(self, parts: Dict, category: str) -> None:
        """
        Checks if the data are not empty and have a train set
        Args:
            parts: train, val and test sets
            category: a grammatical value
        """
        if len(parts) == 3:
            tr_categories_set = set(parts["tr"][1])
            val_categories_set = set(parts["va"][1])
            te_categories_set = set(parts["te"][1])
            if tr_categories_set != val_categories_set:
                logger.warning(
                    f'The classes in train and validation parts are different for category "{category}"'
                )
            elif val_categories_set != te_categories_set:
                logger.warning(
                    f'The classes in train and test parts are different for category "{category}"'
                )

    def subsamples_split(
        self,
        probing_data: List[Tuple[str, str]],
        partition: List[float],
        random_seed: int,
        split: List[str],
    ) -> Dict:
        """
        Splits data into three sets: train, validation, and test
        in the given relation
        Args:
            probing_data: a dictionary that contains grammatical values and
            all the sentences they are found in
            partition: a relation that sentences should be split in
            random_seed: random seed for spliting
            shuffle: if sentences should be randomly shuffled
            split: parts that data should be split to
        """
        parts = {}
        data, labels = map(np.array, zip(*probing_data))
        X_train, X_test, y_train, y_test = train_test_split(
            data,
            labels,
            stratify=labels,
            train_size=partition[0],
            shuffle=self.shuffle,
            random_state=random_seed,
        )

        if len(partition) == 2:
            parts = {split[0]: [X_train, y_train], split[1]: [X_test, y_test]}
        else:
            filtered_labels = self.filter_labels_after_split(y_test)
            if len(filtered_labels) >= 2:
                X_train = X_train[np.isin(y_train, filtered_labels)]
                y_train = y_train[np.isin(y_train, filtered_labels)]
                X_test = X_test[np.isin(y_test, filtered_labels)]
                y_test = y_test[np.isin(y_test, filtered_labels)]

                val_size = partition[1] / (1 - partition[0])
                if y_test.size != 0:
                    X_val, X_test, y_val, y_test = train_test_split(
                        X_test,
                        y_test,
                        stratify=y_test,
                        train_size=val_size,
                        shuffle=self.shuffle,
                        random_state=random_seed,
                    )
                    parts = {
                        split[0]: [X_train, y_train],
                        split[1]: [X_test, y_test],
                        split[2]: [X_val, y_val],
                    }
        return parts

    def generate_probing_file(
        self,
        conllu_text: str,
        category: str,
        splits: List[str],
        partitions: List[float],
        subcategory: str,
        random_seed: int = 42,
    ) -> Dict:
        """
        Generates a split following given arguments
        Args:
            conllu_text: a string in CONLLU format with the data
            category: a grammatical category to split by
            splits: parts that the data are to split to
            partitions: a percentage of splits
            shuffle: if the data should be randomly shuffles
            random_seed: a random seed for spliting
        """
        sentences = parse_tree(conllu_text)
        classified_sentences = self.classify(sentences, category, subcategory)
        num_classes = len(classified_sentences.keys())

        if num_classes == 1:
            logger.warning(f'Category "{category}" has only one class')
            return {}
        elif num_classes == 0:
            logger.warning(
                f'This file does not contain examples of category "{category}"'
            )
            return {}

        if len(splits) == 1:
            data = [
                (s, class_name)
                for class_name, sentences in classified_sentences.items()
                for s in sentences
            ]
            parts = {splits[0]: list(zip(*data))}
            return parts

        data = [
            (s, class_name)
            for class_name, sentences in classified_sentences.items()
            if len(sentences) > num_classes
            for s in sentences
        ]
        if data:
            parts = self.subsamples_split(data, partitions, random_seed, splits)
        else:
            parts = {}

        if not parts:
            logger.warning(
                f'Not enough data of category "{category}" for stratified split'
            )
        return parts

    def get_text_and_categories(
        self, paths: List[os.PathLike]
    ) -> Tuple[List[str], Dict[str, List[Any]]]:
        set_of_values = set()
        subcats: Dict[str, set] = defaultdict(set)
        list_texts = [self.read(str(p)) for p in paths]
        text_data = "\n".join(list_texts)
        token_lists = parse(text_data)
        for token_list in token_lists:
            for token in token_list:
                feats = token["feats"]
                pos = token["upos"]
                deprel = token["deprel"].split(":")[0]

                if self.sorting == "by_pos":
                    if feats and pos:
                        subcats[pos].update(feats.keys())
                elif self.sorting == "by_deprel":
                    if feats and deprel:
                        subcats[deprel].update(feats.keys())
                elif self.sorting == "by_pos_and_deprel":
                    if feats and pos and deprel:
                        subcats[f"{pos}_{deprel}"].update(feats.keys())
                else:
                    if feats:
                        set_of_values.update(feats.keys())

        if not self.sorting:
            subcats["no_sorting"] = set_of_values
        sorted_categories = {key: sorted(value) for key, value in subcats.items()}
        return list_texts, sorted_categories

    def get_filepaths_from_dir(self, dir_path: os.PathLike) -> List[os.PathLike]:
        dir_path = Path(dir_path).resolve() if dir_path is not None else None

        def sorting_parts_func(p: os.PathLike) -> int:
            if "train" in str(p):
                return 0
            elif "dev" in str(p):
                return 1
            return 2

        filepaths = [
            Path(dir_path, p)
            for p in os.listdir(dir_path)
            if re.match(".*-(train|dev|test).*\.conllu", p)
        ]
        return sorted(filepaths, key=sorting_parts_func)

    def __extract_lang_from_udfile_path(
        self, ud_file_path: os.PathLike, language: Optional[str]
    ) -> str:
        if not language:
            return Path(ud_file_path).stem.split("-")[0]
        return language

    def __determine_ud_savepath(
        self, path_from_files: os.PathLike, save_path_dir: Optional[os.PathLike]
    ) -> Path:
        final_path = None
        if not save_path_dir:
            final_path = path_from_files
        else:
            final_path = save_path_dir
        os.makedirs(final_path, exist_ok=True)
        return Path(final_path)

    def prepare_data_for_probing(
        self,
        categories: List[Any],
        list_texts,
        splits,
        partitions,
        subcategory,
    ):
        data: Dict[str, Dict] = defaultdict(dict)
        for category_name in categories:
            category_parts: Dict = {}
            for text, split, part in zip(list_texts, splits, partitions):
                process_part = self.generate_probing_file(
                    conllu_text=text,
                    splits=split,
                    partitions=part,
                    category=category_name,
                    subcategory=subcategory,
                )

                # means that some part within tr, va, te wasn't satisfied to the conditions
                if process_part == {}:
                    category_parts = {}
                    break
                category_parts.update(process_part)

            if category_parts:
                self.check_parts(category_parts, category_name)

            data[f"{subcategory}_{category_name}"] = category_parts

        return data

    def generate_data_by_categories(
        self,
        paths: List[os.PathLike],
        partitions: Optional[List[List[float]]] = None,
        splits: Optional[List[List[str]]] = None,
    ) -> Dict[str, Dict[str, List[str]]]:
        """
        Generates files for all categories
        Args:
            paths: files with data in CONLLU format
            splits: the way how the data should be splitted
            partitions: the percentage of different splits
        """
        data: Dict[str, Dict] = defaultdict(dict)
        list_texts, categories = self.get_text_and_categories(paths)

        if self.verbose:
            num_categories = sum([len(value) for value in categories.values()])
            print(f"{num_categories} categories were found")

        if len(categories) == 0:
            paths_str = "\n".join([str(p) for p in paths])
            logger.warning(
                f"Something went wrong during processing files. None categories were found for paths:\n{paths_str}"
            )

        if partitions is None:
            partitions = partitions_by_files[len(paths)]
        if splits is None:
            splits = splits_by_files[len(paths)]

        for category, values in categories.items():
            print(f"Collecting data for category={category}")
            data.update(
                self.prepare_data_for_probing(
                    values, list_texts, splits, partitions, category
                )
            )

        return data

    @typing.no_type_check
    def process_paths(
        self,
        tr_path: Optional[os.PathLike] = None,
        va_path: Optional[os.PathLike] = None,
        te_path: Optional[os.PathLike] = None,
        language: Optional[str] = None,
        save_path_dir: Optional[os.PathLike] = None,
    ) -> Tuple[Dict[str, Dict[str, List[str]]], str, os.PathLike]:
        known_paths = [Path(p) for p in [tr_path, va_path, te_path] if p is not None]
        assert len(known_paths) > 0, "None paths were provided"
        assert tr_path is not None, "At least the path to train data should be provided"

        language = self.__extract_lang_from_udfile_path(known_paths[0], language)
        save_path_dir = self.__determine_ud_savepath(
            Path(known_paths[0]).parent, save_path_dir
        )

        if len(known_paths) == 1:
            files_data = self.generate_data_by_categories(paths=[tr_path])
        elif len(known_paths) == 2:
            second_path = te_path if te_path is not None else va_path
            files_data = self.generate_data_by_categories(paths=[tr_path, second_path])
        elif len(known_paths) == 3:
            files_data = self.generate_data_by_categories(
                paths=[tr_path, va_path, te_path]
            )
        else:
            raise NotImplementedError(too_much_files_err_str.format(len(known_paths)))
        return files_data, language, save_path_dir

    def convert(
        self,
        tr_path: Optional[os.PathLike] = None,
        va_path: Optional[os.PathLike] = None,
        te_path: Optional[os.PathLike] = None,
        path_dir_conllu: Optional[os.PathLike] = None,
        language: Optional[str] = None,
        save_path_dir: Optional[os.PathLike] = None,
    ) -> None:
        """
        Converts files in CONLLU format to SentEval probing files
        Args:
            tr_path: a path to a file with all data OR train data
            te_path: a path to a file with test data
            va_path: a path to a file with test data
            dir_path: a path to a directory with all files
        """

        if self.verbose:
            paths_str = "\n".join(
                [
                    str(p)
                    for p in [tr_path, va_path, te_path, path_dir_conllu]
                    if p is not None
                ]
            )
            print(f"In progress data from path: {paths_str}")

        if path_dir_conllu:
            paths = [Path(p) for p in self.get_filepaths_from_dir(path_dir_conllu)]
            assert len(paths) > 0, f"Empty folder: {path_dir_conllu}"
            assert len(paths) <= 3, too_much_files_err_str.format(len(paths))
            data, language, save_path_dir = self.process_paths(
                *paths, language=language, save_path_dir=save_path_dir
            )
        else:
            data, language, save_path_dir = self.process_paths(
                tr_path, va_path, te_path, language, save_path_dir
            )

        for category, category_data in data.items():
            if category_data:
                output_path = self.writer(
                    category_data, category, language, save_path_dir  # type: ignore
                )
                if self.verbose:
                    print(f"Results are saved by the path: {output_path}")
