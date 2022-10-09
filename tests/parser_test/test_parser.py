import os
import unittest
from collections import defaultdict
from pathlib import Path

import pytest
from conllu import parse_tree

from probing.ud_parser.ud_parser import ConlluUDParser


@pytest.mark.ud_parser
class TestUDParser(unittest.TestCase):
    path_testfile1 = Path(Path(__file__).parent.resolve(), "test.conllu")
    text_testfile1 = open(path_testfile1, encoding="utf-8").read()

    path_testfile2 = Path(Path(__file__).parent.resolve(), "constructed_test.conllu")
    text_testfile2 = open(path_testfile2, encoding="utf-8").read()

    def test_categories(self):
        parser = ConlluUDParser()
        categories = sorted(
            [
                "Animacy",
                "Case",
                "Gender",
                "Number",
                "Degree",
                "Aspect",
                "Tense",
                "VerbForm",
                "Voice",
                "PronType",
                "Person",
                "Mood",
                "Variant",
                "Poss",
            ]
        )
        texts, found_categories = parser.get_text_and_categories([self.path_testfile1])
        self.assertEqual(categories, found_categories)
        self.assertEqual(list, type(found_categories))

    def test_classify(self):
        parser = ConlluUDParser()
        person_dict = defaultdict(list)
        person_dict["3"].extend(
            [
                "Это связано с тем , что работа каких - то инструкций алгоритма может быть зависима от других инструкций или результатов их работы .",
                "Таким образом , некоторые инструкции должны выполняться строго после завершения работы инструкций , от которых они зависят .",
                "Независимые инструкции или инструкции , ставшие независимыми из - за завершения работы инструкций , от которых они зависят , могут выполняться в произвольном порядке , параллельно или одновременно , если это позволяют используемые процессор и операционная система .",
                "Независимые инструкции или инструкции , ставшие независимыми из - за завершения работы инструкций , от которых они зависят , могут выполняться в произвольном порядке , параллельно или одновременно , если это позволяют используемые процессор и операционная система .",
            ],
        )
        sentences = parse_tree(self.text_testfile1)
        self.assertEqual(person_dict, parser.classify(sentences, "Person"))

    def test_check(self):
        parser = ConlluUDParser()
        category = "Animacy"
        language = "russian"
        save_path_dir = Path("")

        set_2 = {
            "tr": [[1, 2], ["a", "b"]],
            "va": [[1], ["a"]],
            "te": [[1, 2], ["a", "b"]],
        }
        with self.assertLogs("", "DEBUG") as experiment_2:
            answer = parser.check_parts(set_2, category)
        log_2 = f'The classes in train and validation parts are different for category "{category}"'

        set_3 = {
            "tr": [[1, 2], ["a", "b"]],
            "va": [[1], ["a", "b"]],
            "te": [[1, 2], ["a"]],
        }
        with self.assertLogs("", "DEBUG") as experiment_3:
            answer = parser.check_parts(set_3, category)
        log_3 = f'The classes in train and test parts are different for category "{category}"'

        self.assertIn(log_2, experiment_2.output[0])
        self.assertIn(log_3, experiment_3.output[0])

    def test_writer(self):
        parser = ConlluUDParser()
        set_1 = {
            "tr": [[1, 2], ["a", "b"]],
            "va": [[1], ["a"]],
            "te": [[1, 2], ["a", "b"]],
        }
        result_path = parser.writer(set_1, "animacy", "ru", "")
        self.assertIn(result_path.name, os.listdir())
        os.remove(result_path)

    def test_find_tokens(self):
        parser = ConlluUDParser()
        sentences = parse_tree(self.text_testfile1)
        animacy_token = parser.find_category_token(
            category="Animacy", head=sentences[0].token, children=sentences[0].children
        )
        degree_token = parser.find_category_token(
            category="Degree", head=sentences[0].token, children=sentences[0].children
        )
        prontype_token = parser.find_category_token(
            category="PronType", head=sentences[0].token, children=sentences[0].children
        )
        self.assertEqual(
            "набор", animacy_token["form"]
        )  # the root has the category in search
        self.assertEqual(
            "точный", degree_token["form"]
        )  # one of the children has the category in seatch
        self.assertEqual(
            None, prontype_token
        )  # this sentence does not have the category in search

    def test_subsamples(self):
        parser = ConlluUDParser()
        probing_data = [
            ("повторяет", "3"),
            ("повторяешь", "2"),
            ("бывает", "3"),
            ("говорит", "3"),
            ("весит", "3"),
            ("узнаешь", "2"),
        ]

        self.assertEqual(
            {},
            parser.subsamples_split(
                probing_data, partition=[0.8, 0.1, 0.1], random_seed=0, split=["tr", "va", "te"],
            ),
        )
        self.assertEqual(
            2,
            len(
                parser.subsamples_split(
                    probing_data,
                    partition=[0.8, 0.2],
                    split=["tr", "te"],
                    random_seed=0,
                )["te"]
            ),
        )
        self.assertEqual(
            {},
            parser.subsamples_split(
                probing_data + probing_data,
                partition=[0.8, 0.1, 0.1],
                split=["tr", "va", "te"],
                random_seed=0,
            ),
        )

    def test_generate_probing_file(self):
        parser = ConlluUDParser()
        log_1 = 'Category "Degree" has only one class'
        with self.assertLogs("", "DEBUG") as experiment_1:
            parts_1 = parser.generate_probing_file(
                self.text_testfile1,
                "Degree",
                splits=["tr", "va", "te"],
                partitions=[0.8, 0.1, 0.1],
            )

        parts_2 = parser.generate_probing_file(
            self.text_testfile1, "Number", splits=["tr"], partitions=[1.0]
        )

        log_2 = 'Not enough data of category "Voice" for stratified split'
        with self.assertLogs("", "DEBUG") as experiment_2:
            parts_3 = parser.generate_probing_file(
                self.text_testfile2,
                "Voice",
                splits=["tr", "va", "te"],
                partitions=[0.8, 0.1, 0.1],
            )

        log_3 = 'This file does not contain examples of category "Number"'
        with self.assertLogs("", "DEBUG") as experiment_3:
            parts_4 = parser.generate_probing_file(
                self.text_testfile2,
                "Number",
                splits=["tr", "va", "te"],
                partitions=[0.8, 0.1, 0.1],
            )

        cases = ["Nom"]
        parts_5 = parser.generate_probing_file(
            self.text_testfile1, "Case", splits=["va", "te"], partitions=[0.5, 0.5]
        )

        parts_6 = parser.generate_probing_file(
            self.text_testfile2, "Number", splits=["va", "te"], partitions=[0.5, 0.5]
        )

        self.assertIn(log_1, experiment_1.output[0])
        self.assertEqual({}, parts_1)
        self.assertEqual(6, len(parts_2["tr"][0]))
        self.assertIn(log_2, experiment_2.output[0])
        self.assertEqual({}, parts_3)
        self.assertEqual({}, parts_4)
        self.assertIn(log_3, experiment_3.output[0])
        self.assertEqual(set(cases), set(parts_5["te"][1]))
        self.assertEqual({}, parts_6)

    def test_generate1(self):
        parser = ConlluUDParser()
        data = parser.generate_data_by_categories(
            paths=[self.path_testfile1],
            splits=(["tr", "va", "te"],),
            partitions=([0.8, 0.1, 0.1],),
        )
        self.assertEqual(14, len(data.keys()))
        self.assertEqual(
            [
                {},
            ]
            * 14,
            list(data.values()),
        )

    def test_generate2(self):
        parser = ConlluUDParser(verbose=False)

        folder_path = Path(__file__).resolve().parent
        test_file_path = Path(folder_path, "hi_pud-ud-test.conllu")
        data = parser.generate_data_by_categories(paths=[test_file_path])
        self.assertEqual(len(data), 17)

        empty_cats = [
            "Foreign",
            "Mood",
            "NumType",
            "Polarity",
            "Polite",
            "PronType",
            "VerbForm",
        ]
        notempty_cats = [
            "Animacy",
            "Aspect",
            "Definite",
            "Gender",
            "Gender[psor]",
            "Number",
            "Number[psor]",
            "Person",
            "Tense",
        ]
        for c in empty_cats:
            print(c)
            self.assertEqual(data[c], {})

        for c in notempty_cats:
            print(c)
            self.assertTrue(data[c] != {})
