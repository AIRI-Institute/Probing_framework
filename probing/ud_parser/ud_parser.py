import re
import os
import csv
import numpy as np
import logging
from collections import Counter
from enum import Enum
from pathlib import Path
from typing import Tuple, Optional, Dict, List, Any
from collections import defaultdict
from conllu import parse_tree, parse
from conllu.models import TokenTree, Token
from sklearn.model_selection import train_test_split
from nltk.tokenize import wordpunct_tokenize

from probing.ud_parser.ud_config import partitions_by_files, too_much_files_err_str


class ConlluUDParser:
    def __init__(
        self,
        shuffle: bool = True
    ):
        self.shuffle = shuffle

    def read(self, path: str) -> str:
        """
        Reads a file
        Args:
            path: a path to a file
        """
        with open(path, encoding='utf-8') as f:
            conllu_file = f.read()
        return conllu_file 

    def writer(self, partition_sets: Dict, save_path_dir: os.PathLike, category: Enum, language: str) -> Path:
        """
        Writes to a file
        Args:
             result_path: a filename that will be generated
             partition_sets: the data split into 3 parts
        """
        result_path = Path(Path(save_path_dir).resolve(), f'{language}_{category}.csv')
        print(f'Writing to file: {result_path}\n')
        with open(result_path, 'w', encoding='utf-8') as newf:
            my_writer = csv.writer(newf, delimiter='\t', lineterminator='\n')
            for part in partition_sets:
                for sentence, value in zip(*partition_sets[part]):
                    my_writer.writerow([part, value, sentence])
        return result_path

    def find_category_token(
        self,
        category: Enum,
        head: Token,
        children: List[TokenTree]
    ) -> Optional[str]:
        """
        Finds a token that has a given category and is located on the top of a tree
        Args:
            category: a grammatical value that needed to be found
            head: the root of a sentence
            children: children of a token in token tree
        """
        if head['feats'] and category in head['feats']:
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
        category: Enum
    ) -> Dict:
        """
        Classifies sentences by a grammatical value they contain
        Args:
            token_trees: sentences represented as trees
            category: a grammatical value
        """
        probing_data = defaultdict(list)
        for token_tree in token_trees:
            s_text = ' '.join(wordpunct_tokenize(token_tree.metadata['text']))
            root = token_tree.token
            category_token = self.find_category_token(category, root, token_tree.children)
            if category_token:
                value = category_token['feats'][category]
                probing_data[value].append(s_text)
        return probing_data

    def filter_labels_after_split(self, labels: List[Any]) -> List[Any]:
        labels_repeat_dict = Counter(labels)
        n_repeat = 1 # threshold to overcome further splitting problem
        return [label for label, count in labels_repeat_dict.items() if count > n_repeat]

    def check_parts(
        self,
        parts: Dict,
        category: Enum
    ) -> bool:
        """
        Checks if the data are not empty and have a train set
        Args:
            parts: train, val and test sets
            category: a grammatical value
        """
        if len(parts) == 3:
            tr_categories_set = set(parts['tr'][1])
            val_categories_set = set(parts['va'][1])
            te_categories_set = set(parts['te'][1])
            if tr_categories_set != val_categories_set:
                logging.warning(f"The classes in train and validation parts are different for category \"{category}\"")
            elif val_categories_set != te_categories_set:
                logging.warning(f"The classes in train and test parts are different for category \"{category}\"")
            return True
        return False

    def subsamples_split(
        self,
        probing_data: Dict,
        partition: List[float],
        random_seed: int,
        split: List[Enum] = None
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
            data, labels, stratify=labels, train_size=partition[0],
            shuffle=self.shuffle, random_state=random_seed
        )

        if len(partition) == 2:
            parts = {
                split[0]: [X_train, y_train],
                split[1]: [X_test, y_test]
            }
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
                        X_test, y_test, stratify=y_test, train_size=val_size,
                        shuffle=self.shuffle, random_state=random_seed
                    )
                    parts = {
                        split[0]: [X_train, y_train],
                        split[1]: [X_test, y_test],
                        split[2]: [X_val, y_val]
                    }
        return parts

    def generate_probing_file(
        self,
        conllu_text: str,
        category: Enum,
        splits: List[Enum],
        partitions: List[float],
        random_seed: int = 42
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
        classified_sentences = self.classify(sentences, category)
        num_classes = len(classified_sentences.keys())

        if num_classes == 1:
            logging.warning(f"Category \"{category}\" has only one class")
            return {}
        elif num_classes == 0:
            logging.warning(f"This file does not contain examples of category \"{category}\"")
            return {}

        if len(splits) == 1:
            data = [(s, class_name) for class_name, sentences in classified_sentences.items() for s in sentences]
            parts = {splits[0]: list(zip(*data))}
            return parts

        data = [(s, class_name) for class_name, sentences in classified_sentences.items() if len(sentences) > num_classes for s in sentences]
        if data:
            parts = self.subsamples_split(data, partitions, random_seed, splits)
        else:
            parts = {}
        
        if not parts:
            logging.warning(f"Not enough data of category \"{category}\" for stratified split")
        return parts

    def get_text_and_categories(self, paths: List[os.PathLike]) -> Tuple[List[str], List[Enum]]:
        set_of_values = set()
        list_texts = [self.read(p) for p in paths]
        text_data = "\n".join(list_texts)
        token_lists = parse(text_data)
        for token_list in token_lists:
            for token in token_list:
                feats = token['feats']
                if feats:
                    set_of_values.update(feats.keys())
        return list_texts, sorted(set_of_values)
    
    def get_filepaths_from_dir(self, dir_path: os.PathLike) -> List[os.PathLike]:
        dir_path = Path(dir_path).resolve() if dir_path is not None else None 
        def sorting_parts_func(p: os.PathLike) -> int:
            p = str(p)
            if 'train' in p:
                return 0
            elif 'dev' in p:
                return 1
            return 2

        filepaths = [Path(dir_path, p) for p in os.listdir(dir_path) if \
                re.match(".*-(train|dev|test).*\.conllu", p)]
        return sorted(filepaths, key=sorting_parts_func)

    def __extract_lang_from_udfile_path(
        self,
        ud_file_path: os.PathLike,
        language: Optional[str]
    ) -> str:
        if not language:
            return Path(ud_file_path).stem.split('-')[0]
        return language
    
    def __determine_ud_savepath(
        self,
        path_from_files: os.PathLike,
        save_path_dir: Optional[os.PathLike]
    ) -> Path:
        final_path = None
        if not save_path_dir:
            final_path = path_from_files
        else:
            final_path = save_path_dir
        os.makedirs(final_path, exist_ok=True)
        return Path(final_path)

    def generate_data_by_categories(
        self,
        paths: List[os.PathLike],
        splits: List[List[Enum]],
        partitions: List[List[float]],
        language: Optional[str],
        save_path_dir: Optional[os.PathLike]
    ) -> Dict[str, Dict[Enum, List[str]]]:
        """
        Generates files for all categories
        Args:
            paths: files with data in CONLLU format
            splits: the way how the data should be split
            partitions: the percentage of different splits
        """
        data = defaultdict(dict)
        language = self.__extract_lang_from_udfile_path(paths[0], language)
        save_path_dir = self.__determine_ud_savepath(Path(paths[0]).parent, save_path_dir)
        list_texts, categories = self.get_text_and_categories(paths)

        if len(categories) == 0:
            paths_str = "\n".join([str(p) for p in paths])
            logging.warning(f"Something went wrong during processing files. None categories were found for paths:\n{paths_str}")

        for category in categories:
            category_parts = {}
            for text, split, part in zip(list_texts, splits, partitions):
                part = self.generate_probing_file(
                    conllu_text=text, splits=split,
                    partitions=part, category=category
                )
                category_parts.update(part)

            data[category] = category_parts
            are_full_parts = self.check_parts(category_parts, category)
            if are_full_parts:
                output_path = self.writer(category_parts, save_path_dir, category, language)
        return data

    def process_paths(
        self,
        tr_path: os.PathLike = None,
        va_path: os.PathLike = None,
        te_path: os.PathLike = None,
        language: Optional[str] = None,
        save_path_dir: Optional[os.PathLike] = None
    ) -> None:
        known_paths = [Path(p) for p in [tr_path, va_path, te_path] if p is not None]
        assert len(known_paths) > 0, "None paths were provided"
        assert tr_path is not None, "At least the path to train data should be provided"

        if len(known_paths) == 1:
            _ = self.generate_data_by_categories(
                paths=[tr_path],
                partitions=partitions_by_files["one_file"],
                splits = [["tr", "va", "te"]],
                language = language,
                save_path_dir = save_path_dir
            )
        elif len(known_paths) == 2:
            second_path = te_path if te_path is not None else va_path
            _ = self.generate_data_by_categories(
                paths=[tr_path, second_path],
                partitions=partitions_by_files["two_files"],
                splits = [["tr"], ["va", "te"]],
                language = language,
                save_path_dir = save_path_dir
            )
        elif len(known_paths) == 3:
            _ = self.generate_data_by_categories(
                paths=[tr_path, va_path, te_path],
                partitions=partitions_by_files["three_files"],
                splits = [["tr"], ["va"], ["te"]],
                language = language,
                save_path_dir = save_path_dir
            )
        else:
            raise NotImplementedError(too_much_files_err_str.format(len(known_paths)))

    def convert(
        self,
        tr_path: Optional[os.PathLike] = None,
        va_path: Optional[os.PathLike] = None,
        te_path: Optional[os.PathLike] = None,
        dir_conllu_path: Optional[os.PathLike] = None,
        language: Optional[str] = None,
        save_path_dir: Optional[os.PathLike] = None
    ) -> None:
        """
        Converts files in CONLLU format to SentEval probing files
        Args:
            tr_path: a path to a file with all data OR train data
            te_path: a path to a file with test data
            va_path: a path to a file with test data
            dir_path: a path to a directory with all files
        """
        if dir_conllu_path:
            paths = [Path(p) for p in self.get_filepaths_from_dir(dir_conllu_path)]
            assert len(paths) > 0, f"Empty folder: {dir_conllu_path}"
            assert len(paths) <= 3, too_much_files_err_str.format(len(paths))
            self.process_paths(*paths, language=language, save_path_dir=save_path_dir)
        else:
            self.process_paths(tr_path, va_path, te_path, language, save_path_dir)
