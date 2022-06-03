from ud_parser import Splitter

import pytest
import os
import unittest
from conllu import parse, parse_tree
from collections import defaultdict
from pathlib import Path


@pytest.mark.ud_parser
class TestUDParser(unittest.TestCase):

    def test_categories(self):
        parser = Splitter()
        categories = sorted(["Animacy", "Case", "Gender", "Number", "Degree", "Aspect", "Tense",
                             "VerbForm", "Voice", "PronType", "Person", "Mood", "Variant", "Poss"])
        conllu_text = open("test.conllu", encoding="utf-8").read()
        found_categories = parser.find_categories(conllu_text)
        self.assertEqual(categories, found_categories)
        self.assertEqual(list, type(found_categories))

    def test_classify(self):
        parser = Splitter()
        person_dict = defaultdict(list)
        person_dict["3"].extend(["Это связано с тем , что работа каких - то инструкций алгоритма может быть зависима от других инструкций или результатов их работы .",
        "Таким образом , некоторые инструкции должны выполняться строго после завершения работы инструкций , от которых они зависят .",
        "Независимые инструкции или инструкции , ставшие независимыми из - за завершения работы инструкций , от которых они зависят , могут выполняться в произвольном порядке , параллельно или одновременно , если это позволяют используемые процессор и операционная система .",
        "Независимые инструкции или инструкции , ставшие независимыми из - за завершения работы инструкций , от которых они зависят , могут выполняться в произвольном порядке , параллельно или одновременно , если это позволяют используемые процессор и операционная система ."],)
        sentences = parse_tree(open("test.conllu", encoding="utf-8").read())
        self.assertEqual(person_dict, parser.classify(sentences, "Person"))

    def test_check(self):
        parser = Splitter()
        category = "Animacy"
        set_1 = {"tr": [], "te": ["a"], "va": ["b"]}
        with self.assertLogs('', 'DEBUG') as experiment_1:
            parser.check(set_1, category)
        log_1 = f'WARNING:root:One of the files does not contain examples for {category} \n'

        set_2 = {"tr": [[1, 2], ["a", "b"]],
                 "va": [[1], ["a"]],
                 "te": [[1, 2], ["a", "b"]], }
        parser.save_path_dir = Path("")
        parser.language = "russian"
        with self.assertLogs('', 'DEBUG') as experiment_2:
            parser.check(set_2, category)
        log_2 = "WARNING:root:The number of category meanings is different in train and validation parts."

        set_3 = {"tr": [[1, 2], ["a", "b"]],
                 "va": [[1], ["a", "b"]],
                 "te": [[1, 2], ["a"]], }
        with self.assertLogs('', 'DEBUG') as experiment_3:
            parser.check(set_3, category)
        log_3 = "WARNING:root:The number of category meanings is different in train and test parts."

        with self.assertLogs('', 'DEBUG') as experiment_4:
            parser.check({}, category)
        log_4 = f"WARNING:root:There are no examples for {category} in this language \n"

        self.assertEqual(log_1, experiment_1.output[0])
        self.assertEqual(log_2, experiment_2.output[0])
        self.assertEqual(log_3, experiment_3.output[0])
        self.assertEqual(log_4, experiment_4.output[0])

    def test_writer(self):
        parser = Splitter()
        result_path = "ru_animacy.csv"
        set_1 = {"tr": [[1, 2], ["a", "b"]],
                 "va": [[1], ["a"]],
                 "te": [[1, 2], ["a", "b"]], }
        parser.writer(result_path, set_1)
        self.assertIn(result_path, os.listdir())

    def test_find_tokens(self):
        parser = Splitter()
        text = open("test.conllu", encoding="utf-8").read()
        sentences = parse_tree(text)
        animacy_token = parser.find_category_token(category="Animacy",
                                                   head=sentences[0].token,
                                                   children=sentences[0].children)
        degree_token = parser.find_category_token(category="Degree",
                                                  head=sentences[0].token,
                                                  children=sentences[0].children)
        prontype_token = parser.find_category_token(category="PronType",
                                                    head=sentences[0].token,
                                                    children=sentences[0].children)
        self.assertEqual("набор", animacy_token["form"])  # the root has the category in search
        self.assertEqual("точный", degree_token["form"])   # one of the children has the category in seatch
        self.assertEqual(None, prontype_token)  # this sentence does not have the category in search

    def test_subsamples(self):
        parser = Splitter()
        probing_data = [("повторяет", "3"), ("повторяешь", "2"),
                        ("бывает", "3"), ("говорит", "3"),
                        ("весит", "3"), ("узнаешь", "2")]

        self.assertEqual({}, parser.subsamples_split(probing_data,
                                                     partition=[0.8, 0.1, 0.1],
                                                     random_seed=0))
        self.assertEqual(2, len(parser.subsamples_split(probing_data,
                                                        partition=[0.8, 0.2],
                                                        split=["tr", "te"],
                                                        random_seed=0)["te"]))
        self.assertEqual(2, len(parser.subsamples_split(probing_data + probing_data,
                                                        partition=[0.8, 0.1, 0.1],
                                                        split=["tr", "va", "te"],
                                                        random_seed=0)["te"]))

    def test_generate_probing_file(self):
        parser = Splitter()
        conllu_text = open("test.conllu", encoding="utf-8").read()
        log_1 = "WARNING:root:Category Degree has one value"
        with self.assertLogs('', 'DEBUG') as experiment_1:
            parts_1 = parser.generate_probing_file(conllu_text, "Degree")

        parts_2 = parser.generate_probing_file(conllu_text, "Number", splits=["tr"])
        parts_3 = parser.generate_probing_file(conllu_text, "Number", splits=["tr", "va", "te"])

        self.assertEqual(log_1, experiment_1.output[0])
        self.assertEqual({}, parts_1)
        self.assertEqual(6, len(parts_2["tr"][0]))
        self.assertEqual({}, parts_3)

    def test_generate(self):
        parser = Splitter()
        data = parser.generate_(paths=["test.conllu"], splits=(["tr", "va", "te"],), partitions=([0.8, 0.1, 0.1],))
        self.assertEqual(14, len(data.keys()))
        self.assertEqual([{}, ] * 14, list(data.values()))

    def test_convert(self):
        parser = Splitter()
        self.assertRaises(ValueError, parser.convert, "test.conllu", partitions=[0.9, 0.1, 0.1])


if __name__ == '__main__':
    unittest.main()
#    unittest.main(argv=['first-arg-is-ignored'], exit=False)
