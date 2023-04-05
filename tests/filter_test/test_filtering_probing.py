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

        self.dir_conllu_path = "./test_filter_probing_data/conllu_folder/recl"

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
        conllu_paths = [
            "./test_filter_probing_data/conllu_folder/recl/ru_taiga-ud-test1.conllu",
            "./test_filter_probing_data/conllu_folder/recl/ru_taiga-ud-test2.conllu",
        ]
        self.probing_filter.upload_files(conllu_paths=conllu_paths)
        self.assertEqual("ru_taiga", self.probing_filter.language)
        self.assertEqual(32, len(self.probing_filter.sentences))

    def test_upload_files_dir_conllu_path(self):
        self.probing_filter.upload_files(dir_conllu_path=self.dir_conllu_path)
        self.assertEqual("ru_taiga", self.probing_filter.language)
        self.assertEqual(32, len(self.probing_filter.sentences))

    def test_upload_files_empty_folder(self):
        empty_path = "./test_filter_probing_data/empty_folder"
        with self.assertRaises(AssertionError):
            self.probing_filter.upload_files(dir_conllu_path=empty_path)

    def test_upload_files_not_conllu_folder(self):
        not_conllu_path = "./test_filter_probing_data/not_conllu_folder"
        with self.assertRaises(AssertionError):
            self.probing_filter.upload_files(dir_conllu_path=not_conllu_path)

    def test_upload_files_no_paths(self):
        with self.assertRaises(Exception):
            self.probing_filter.upload_files()

    def test_filter_and_convert_all_saved(self):
        queries_sents = {
            "ADVCL": [
                "Она решила попытаться остановить машину — хотя выйдя под дождь , сразу же промокла насквозь .",
                "И охота завыть , вскинув морду к луне .",
                "И не предложит выпить , если ты решил жить трезвым .",
                "Смерть твоя — настолько благая весть , что посовестись — и умри !",
                "Ну , ложись им под ноги , в прах ложись , потому что уже пора !",
                "В печали ль , в радости ль , во хмелю , в потемках земельных недр , Я вас всей кровью своей люблю , сады мои — метр на метр !",
                "Как защитить их , себя казня , до жуткой храня поры ?",
                "Как сообщается в пресс - релизе университета , программу можно использовать на любом смартфоне .",
                "Вячеслав , почему бы Вам не возглавить КПРФ Пока оно ещё есть .",
                "Если ты устал , иди спать .",
                "Если ты голодный , иди есть .",
                "Когда на улице темно , надо быть осторожным .",
                "Когда друзья тебя не слышат , не надо быть настойчивым .",
                "Мы всё сдадим , потому что мы хорошие студенты .",
            ],
            "ACL": [
                "Счастье это качество , не имеющее будущего и прошлого .",
                "Но есть мужчина , которого я не хотела бы потерять ...",
                "Среди разных сыновей был один , который звал себя Сыном Божьим .",
                "Неужто вправду сгорел тот мост , которым я к ним пройду ?!",
                "Она заставляет смартфон постоянно испускать высокочастотный звук , неразличимый для человеческого уха , но улавливаемый микрофоном устройства .",
                "То , что никакого отношения к ним не имеет",
                'Депутат ЛДПР , которого не пустили в " Европейский ", объяснил причину конфликта с охранниками',
                "И пусть всё то , что кажется так сложно , решается красиво и легко !",
                "Пришел мальчик , которому мама не дает конфеты .",
                "Я увидел девочку , которая очень хочет спать .",
                "Я увидел женщину , которую показывали в новостях .",
                "Мальчик взял игрушку , с которой не расставался с самого рождения .",
                "Девочка съела кашу , которую для нее приготовил папа .",
            ],
        }

        task_dir = TemporaryDirectory()
        self.probing_filter.upload_files(dir_conllu_path=self.dir_conllu_path)
        self.probing_filter.filter_and_convert(
            queries=self.queries,
            save_dir_path=task_dir.name,
            task_name="cl",
        )
        self.assertEqual(queries_sents, self.probing_filter.probing_dict)
        with open(f"{task_dir.name}/ru_taiga_cl.csv") as f:
            self.assertEqual(27, len(f.readlines()))
        task_dir.cleanup()

    def test_filter_and_convert_too_few_sentences(self):
        self.probing_filter.upload_files(
            dir_conllu_path="./test_filter_probing_data/conllu_folder/recl_too_few"
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
