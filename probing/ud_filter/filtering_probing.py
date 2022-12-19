import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from conllu import parse
from nltk.tokenize import wordpunct_tokenize

from probing.ud_filter.sentence_filter import SentenceFilter
from probing.ud_filter.utils import (
    delete_duplicates,
    determine_ud_savepath,
    extract_lang_from_udfile_path,
    read,
    subsamples_split,
    writer,
)


class ProbingConlluFilter:
    """

    Creates a probing task based on user's query to a conllu file

    Attributes:
        paths: list of paths to conllu files
        sentences: list of sentences represented as TokenList
        classes: probing task classes based on user's queries {class_label: (node_pattern, constraints)}
        probing_dict: dictionary of classified sentences {class_label: [sentences]}
        parts_data: dictionary of classified sentences divided into tr, val, te parts {part: [[sentences], [labels]]}

    """

    def __init__(self, shuffle: bool = True):

        self.shuffle = shuffle
        self.paths = None
        self.language = None
        self.sentences = []
        self.classes = None
        self.probing_dict = None
        self.parts_data = None

    def upload_files(
        self,
        conllu_paths: List[Optional[os.PathLike]] = None,
        dir_conllu_path: Optional[os.PathLike] = None,
        language: str = None,
    ):
        """Reads and combines conllu files (if there are several), parses them and saves as a list of TokenTree"""

        if dir_conllu_path:
            dir_path = Path(dir_conllu_path).resolve()
            self.paths = [
                Path(dir_path, p)
                for p in os.listdir(dir_path)
                if re.match(r".*\.conllu", p)
            ]
            assert len(self.paths) > 0, f"Empty folder: {dir_conllu_path}"
        elif conllu_paths:
            self.paths = [
                Path(p).resolve() for p in conllu_paths if re.match(r".*\.conllu", p)
            ]
            assert len(self.paths) > 0, f"None of the files are with .conllu extension"
        else:
            raise Exception("pass at least one conllu_path or dir_conllu_path")

        list_texts = [read(p) for p in self.paths]
        conllu_data = "\n".join(list_texts)

        self.language = extract_lang_from_udfile_path(self.paths[0], language=language)
        self.sentences = parse(conllu_data)

    def _filter_conllu(self, class_label: str):
        """Filters sentences by class's query and saves the result to the relevant fields"""

        matching = []
        not_matching = []

        node_pattern = self.classes[class_label][0]
        constraints = self.classes[class_label][1]

        if not self.sentences:
            raise Exception(
                "You haven't uploaded your files yet. Call 'upload_files' with appropriate arguments "
                "before using this method"
            )
        for sentence in self.sentences:
            sf = SentenceFilter(sentence)
            tokenized_sentence = " ".join(wordpunct_tokenize(sentence.metadata["text"]))
            if sf.filter_sentence(node_pattern, constraints):
                matching.append(tokenized_sentence)
            else:
                not_matching.append(tokenized_sentence)
        return matching, not_matching

    def filter_and_convert(
        self,
        queries: Dict[str, Tuple[dict, dict]],
        save_dir_path: Optional[os.PathLike] = None,
        task_name: str = "CustomTask",
        partition: List[float] = [0.8, 0.1, 0.1],
    ):
        """
        Uses user's queries to create a probing task from uploaded conllu files
        Args:
            save_dir_path: where to save a result file (if None saves in the same directory with conllu files
            task_name: name for the probing task (will be used in a result file name)
            partition: a partition for train, validation and test parts
            queries: {class_label: (node_pattern, constraints)}

        Returns:
            path to the result file
        """

        if sum(partition) != 1.0:
            logging.warning(
                f"Your parts in {partition} doesn't add up to 1, so it was automatically changed to {[[0.8, 0.1, 0.1]]}"
            )
            partition = [0.8, 0.1, 0.1]

        self.classes = queries
        self.probing_dict = {}
        for label in self.classes:
            matching, not_matching = self._filter_conllu(label)
            self.probing_dict[label] = matching
            if len(self.classes) == 1:
                self.probing_dict["not_" + list(self.classes.keys())[0]] = not_matching
        self.probing_dict = delete_duplicates(self.probing_dict)

        self.parts_data = subsamples_split(
            self.probing_dict,
            partition=partition,
            random_seed=3,
            shuffle=self.shuffle,
            split=["tr", "va", "te"],
        )

        save_dir = determine_ud_savepath(
            Path(self.paths[0]).parent, save_path_dir=save_dir_path
        )
        output_path = writer(
            self.parts_data, task_name, self.language, save_path_dir=save_dir
        )

        return output_path
