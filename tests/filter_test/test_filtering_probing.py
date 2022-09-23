from probing.ud_filter.filtering_probing import ProbingConlluFilter
from queries import by_passive, SOmatchingNumber, ADPdistance

import pytest
import unittest
from conllu import parse
from pathlib import Path


@pytest.mark.filtering_probing
class TestProbingConlluFilter(unittest.TestCase):
    path_testfile = Path(Path(__file__).parent.resolve(), "sample.conllu")
    text_testfile = open(path_testfile, encoding="utf-8").read()
    trees_testfile = parse(text_testfile)

    def test__filter_conllu(self):
        probing_filter = ProbingConlluFilter()
        probing_filter.sentences = self.trees_testfile
        probing_filter.classes = {'by_passive': by_passive,
                                  'SOmatchingNumber': SOmatchingNumber,
                                  'ADPdistance': ADPdistance}

        by_passive_res = ["I would understand if I was being treated this way by a staff member but the club ' s "
                          "actual OWNER ?!",
                          'Attached for your review are copies of the settlement documents that were filed today in '
                          'the Gas Industry Restructuring / Natural Gas Strategy proceeding , including the Motion '
                          'for Approval of the Comprehensive Settlement that is supported by thirty signatories to '
                          'the Comprehensive Settlement , the Comprehensive Settlement document itself , '
                          'and the various appendices to the settlement .?']
        SOmatchingNumber_res = ['They are kind of in rank order but as I stated if I find the piece that I like we '
                                'will purchase it .',
                                'Masha bought a frying pan , and the boys bought vegetables']
        ADPdistance_res = ['This would have to be determined on a case by case basis .']

        self.assertEqual(probing_filter._filter_conllu('by_passive'), by_passive_res)
        self.assertEqual(probing_filter._filter_conllu('SOmatchingNumber'), SOmatchingNumber_res)
        self.assertEqual(probing_filter._filter_conllu('ADPdistance'), ADPdistance_res)
