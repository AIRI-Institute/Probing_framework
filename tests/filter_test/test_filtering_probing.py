import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, Tuple

import pytest
from conllu import parse
from queries import ADPdistance, SOmatchingNumber, by_passive

from probing.ud_filter.filtering_probing import ProbingConlluFilter


@pytest.mark.filtering_probing
class TestProbingConlluFilter(unittest.TestCase):
    path_testfile = Path(Path(__file__).parent.resolve(), "sample.conllu")
    text_testfile = open(path_testfile, encoding="utf-8").read()
    trees_testfile = parse(text_testfile)

    def setUp(self) -> None:
        self.probing_filter = ProbingConlluFilter()
        self.queries: Dict[
            str, Tuple[Dict[str, Dict[str, str]], Dict[Tuple[str, str], Dict[str, str]]]
        ] = {
            "ADVCL": ({"H": {}, "CL": {}}, {("H", "CL"): {"deprels": "^advcl$"}}),
            "ACL": ({"H": {}, "CL": {}}, {("H", "CL"): {"deprels": "^acl.*?$"}}),
        }

        self.dir_conllu_path = Path(
            Path(__file__).parent.resolve(),
            "test_filter_probing_data/conllu_folder/recl",
        )

    def test__filter_conllu_all_found(self):
        self.probing_filter.sentences = self.trees_testfile
        self.probing_filter.classes = {
            "by_passive": by_passive,
            "SOmatchingNumber": SOmatchingNumber,
            "ADPdistance": ADPdistance,
        }

        by_passive_res = [
            "I would understand if I was being treated this way by a staff member but the club ' s "
            "actual OWNER ?!",
            "Attached for your review are copies of the settlement documents that were filed today in "
            "the Gas Industry Restructuring / Natural Gas Strategy proceeding , including the Motion "
            "for Approval of the Comprehensive Settlement that is supported by thirty signatories to "
            "the Comprehensive Settlement , the Comprehensive Settlement document itself , "
            "and the various appendices to the settlement .?",
        ]
        SOmatchingNumber_res = [
            "They are kind of in rank order but as I stated if I find the piece that I like we "
            "will purchase it .",
            "Masha bought a frying pan , and the boys bought vegetables",
        ]
        ADPdistance_res = ["This would have to be determined on a case by case basis ."]

        self.assertEqual(
            self.probing_filter._filter_conllu("by_passive")[0], by_passive_res
        )
        self.assertEqual(
            self.probing_filter._filter_conllu("SOmatchingNumber")[0],
            SOmatchingNumber_res,
        )
        self.assertEqual(
            self.probing_filter._filter_conllu("ADPdistance")[0], ADPdistance_res
        )

    def test__filter_conllu_no_sentences(self):
        self.probing_filter.sentences = []
        self.probing_filter.classes = {"by_passive": by_passive}

        with self.assertRaises(Exception):
            self.probing_filter._filter_conllu("by_passive")[0]

    def test_upload_files_conllu_paths(self):
        conllu_paths_raw = [
            "test_filter_probing_data/conllu_folder/recl/ru_taiga-ud-test1.conllu",
            "test_filter_probing_data/conllu_folder/recl/ru_taiga-ud-test2.conllu",
        ]
        conllu_paths = [
            str(Path(Path(__file__).parent.resolve(), p)) for p in conllu_paths_raw
        ]
        self.probing_filter.upload_files(conllu_paths=conllu_paths)
        self.assertEqual("ru_taiga", self.probing_filter.language)
        self.assertEqual(32, len(self.probing_filter.sentences))

    def test_upload_files_dir_conllu_path(self):
        self.probing_filter.upload_files(dir_conllu_path=self.dir_conllu_path)
        self.assertEqual("ru_taiga", self.probing_filter.language)
        self.assertEqual(32, len(self.probing_filter.sentences))

    def test_upload_files_empty_folder(self):
        empty_path = Path(
            Path(__file__).parent.resolve(), "test_filter_probing_data/empty_folder"
        )
        with self.assertRaises(AssertionError):
            self.probing_filter.upload_files(dir_conllu_path=empty_path)

    def test_upload_files_not_conllu_folder(self):
        not_conllu_path = Path(
            Path(__file__).parent.resolve(),
            "test_filter_probing_data/not_conllu_folder",
        )
        with self.assertRaises(AssertionError):
            self.probing_filter.upload_files(dir_conllu_path=not_conllu_path)

    def test_upload_files_no_paths(self):
        with self.assertRaises(Exception):
            self.probing_filter.upload_files()

    def test_filter_and_convert_too_few_sentences(self):
        self.probing_filter.upload_files(
            dir_conllu_path=Path(
                Path(__file__).parent.resolve(),
                "test_filter_probing_data/conllu_folder/recl_too_few",
            )
        )
        with self.assertRaises(Exception):
            self.probing_filter.filter_and_convert(queries=self.queries, task_name="cl")

    def test_filter_and_convert_wrong_partitions_sum(self):
        task_dir = TemporaryDirectory()
        partition = [0.9, 0.2, 0.1]
        self.probing_filter.upload_files(dir_conllu_path=self.dir_conllu_path)

        with self.assertLogs() as captured:
            self.probing_filter.filter_and_convert(
                queries=self.queries, save_dir_path=task_dir.name, partition=partition
            )
        self.assertEqual(len(captured.records), 1)
        self.assertEqual(
            captured.records[0].getMessage(),
            "Your parts in [0.9, 0.2, 0.1] doesn't add up to 1, so it was automatically changed to [[0.8, 0.1, 0.1]]",
        )
        task_dir.cleanup()
