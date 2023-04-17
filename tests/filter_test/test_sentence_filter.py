import unittest
from itertools import product
from pathlib import Path

import pytest
from conllu import parse
from queries import ADPdistance, SOmatchingNumber, by_passive

from probing.ud_filter.sentence_filter import SentenceFilter


@pytest.mark.sentence_filter
class TestSentenceFilter(unittest.TestCase):
    path_testfile = Path(Path(__file__).parent.resolve(), "sample.conllu")
    text_testfile = open(path_testfile, encoding="utf-8").read()
    trees_testfile = parse(text_testfile)

    def test_token_match_node(self):
        sf = SentenceFilter(self.trees_testfile[0])
        token = self.trees_testfile[0][5]  # (I) was
        patterns = [
            {"V": {"upos": "AUX"}},  # True
            {"V": {"upos": "VERB"}},  # False
            {"V": {"upos": "^(?!VERB$).*$"}},  # True
            {"V": {"Number": "Sing", "Person": "1"}},  # True
            {"V": {"Number": "Plur", "Person": "1"}},  # False
            {"V": {"Number": "Plur|Sing", "Person": "1"}},  # True
            {"V": {"exclude": ["Definite", "PronType"]}},  # True
            {"V": {"exclude": ["PronType", "Number"]}},  # False
        ]
        results = [sf.token_match_node(token, node_pattern=p["V"]) for p in patterns]
        answers = [True, False, True, True, False, True, True, False]
        self.assertEqual(results, answers)

    def test_search_suitable_tokens(self):
        sent = self.trees_testfile[
            0
        ]  # I would understand if I was being treated this way by a staff member but the club's actual OWNER?!
        sf = SentenceFilter(sent)
        node_pattern = {
            "N": {"Number": "Sing"},  # words in singular number
            "M": {
                "upos": "^(?!NOUN|PRON$).*$",
                "Number": "Sing",
            },  # words in singular number except for pronouns and nouns
            "K": {
                "Number": "Sing",
                "exclude": ["PronType"],
            },  # words in singular number but without 'PronType' category
        }
        answers = {
            "N": ["I", "I", "be", "this", "way", "staff", "member", "club", "owner"],
            "M": ["be", "this"],
            "K": ["be", "way", "staff", "member", "club", "owner"],
        }
        wrong_answers = {
            "N": [
                "I",
                "I",
                "be",
                "this",
                "way",
                "staff",
                "member",
                "club",
                "owner",
                "if",
            ],
            "M": ["be", "this", "club"],
            "K": ["be", "way", "staff", "member", "club", "owner", "I"],
        }
        sf.node_pattern = node_pattern
        sf.nodes_tokens = {node: [] for node in sf.node_pattern}
        for node in node_pattern:
            sf.search_suitable_tokens(node)
        results = {
            n: [sent.filter(id=t + 1)[0]["lemma"] for t in sf.nodes_tokens[n]]
            for n in sf.nodes_tokens
        }
        self.assertEqual(results, answers)
        self.assertNotEqual(results, wrong_answers)

    def test_find_all_nodes_bypassive(self):
        sf_true = SentenceFilter(self.trees_testfile[4])  # sentence with by-passive

        sf_true.node_pattern = by_passive[0]
        sf_true.nodes_tokens = {node: [] for node in sf_true.node_pattern}
        self.assertTrue(sf_true.find_all_nodes())

        sf_false = SentenceFilter(self.trees_testfile[7])  # sentence without by-passive
        sf_false.node_pattern = by_passive[0]
        sf_false.nodes_tokens = {node: [] for node in sf_false.node_pattern}
        self.assertFalse(sf_false.find_all_nodes())

    def test_find_all_nodes_ADPdistance(self):
        sf_true = SentenceFilter(
            self.trees_testfile[4]
        )  # sentence with preposition and some other word

        sf_true.node_pattern = ADPdistance[0]
        sf_true.nodes_tokens = {node: [] for node in sf_true.node_pattern}
        self.assertTrue(sf_true.find_all_nodes())

        sf_false = SentenceFilter(
            self.trees_testfile[7]
        )  # sentence without preposition
        sf_false.node_pattern = ADPdistance[0]
        sf_false.nodes_tokens = {node: [] for node in sf_false.node_pattern}
        self.assertFalse(sf_false.find_all_nodes())

    def test_pattern_relations(self):
        sent = self.trees_testfile[0]
        sf = SentenceFilter(sent)
        sf.sent_deprels = sf.all_deprels()
        patterns = [
            "nsubj(:.*)?",  # nsubj, nsubj:pass
            ".mod(:.*)?",  # amod, nmod:poss
            "nsubj",  # only nsubj,  as «^nsubj$», not «.*nsubj.*»
        ]
        answers = [["nsubj", "nsubj:pass"], ["nmod:poss", "amod"], ["nsubj"]]
        for p, a in zip(patterns, answers):
            self.assertEqual(sf.pattern_relations(p), a)

    def test_pairs_matching_relpattern_answer(self):
        sent = self.trees_testfile[0]
        sf = SentenceFilter(sent)
        relpattern = {("N", "M"): {"deprels": "nsubj(:.*)?"}}
        answer = {(2, 0), (7, 4)}

        sf.nodes_tokens = {
            "N": [i for i in range(len(sent)) if i != 16],
            "M": [i for i in range(len(sent)) if i != 16],
        }
        sf.possible_token_pairs = {
            ("N", "M"): list(product(sf.nodes_tokens["N"], sf.nodes_tokens["M"]))
        }
        sf.sent_deprels = sf.all_deprels()
        sf.constraints = relpattern

        self.assertEqual(answer, sf.pairs_matching_relpattern(("N", "M")))

    def test_pairs_matching_relpattern_not_in_sent_deprels(self):
        sent = self.trees_testfile[0]
        sf = SentenceFilter(sent)
        relpattern = {("N", "M"): {"deprels": "obl:tmod"}}
        answer = set()

        sf.nodes_tokens = {
            "N": [i for i in range(len(sent)) if i != 16],
            "M": [i for i in range(len(sent)) if i != 16],
        }
        sf.possible_token_pairs = {
            ("N", "M"): list(product(sf.nodes_tokens["N"], sf.nodes_tokens["M"]))
        }
        sf.sent_deprels = sf.all_deprels()
        sf.constraints = relpattern

        self.assertNotIn(relpattern[("N", "M")]["deprels"], sf.sent_deprels)
        self.assertEqual(answer, sf.pairs_matching_relpattern(("N", "M")))

    def test_linear_distance(self):
        sent = self.trees_testfile[0]
        sf = SentenceFilter(sent)
        sf.possible_token_pairs = {("N", "M"): [(2, 3), (15, 12), (1, 9)]}
        sf.constraints = {("N", "M"): {"lindist": (-3, 5)}}
        answer = {(2, 3), (15, 12)}
        self.assertEqual(sf.linear_distance(("N", "M")), answer)

    def test_pair_match_fconstraint_true(self):
        sent = self.trees_testfile[1]
        sf = SentenceFilter(sent)
        fconstraints = {"intersec": ["VerbForm", "Number"], "disjoint": ["Tense"]}
        token_pairs = [
            (4, 11),  # matches
            (4, 35),  # Same value for Tense, while should be different
            (11, 35),  # Different value for Number, while should be the same
            (35, 36),  # One token doesn't have pne of the categories
        ]
        answers = [True, False, False, False]
        results = [sf.pair_match_fconstraint(tp, fconstraints) for tp in token_pairs]
        self.assertEqual(answers, results)

    def test_pair_match_fconstraint_wrong_type(self):
        sent = self.trees_testfile[1]
        sf = SentenceFilter(sent)
        fconstraints = {"wrong_fconstraint_type": ["VerbForm", "Number"]}
        token_pairs = [
            (4, 11),  # matches
            (4, 35),  # Same value for Tense, while should be different
            (11, 35),  # Different value for Number, while should be the same
            (35, 36),  # One token doesn't have pne of the categories
        ]

        for tp in token_pairs:
            with self.assertRaises(ValueError):
                sf.pair_match_fconstraint(tp, fconstraints)

    def test_feature_constraint(self):
        sent = self.trees_testfile[1]
        sf = SentenceFilter(sent)
        sf.constraints = {
            ("N", "M"): {
                "fconstraint": {"intersec": ["VerbForm"], "disjoint": ["Tense"]}
            }
        }

        sf.nodes_tokens = {
            "N": [i for i in range(len(sent))],
            "M": [i for i in range(len(sent))],
        }
        sf.possible_token_pairs = {
            ("N", "M"): list(product(sf.nodes_tokens["N"], sf.nodes_tokens["M"]))
        }
        answer = {(4, 11), (11, 4), (11, 35), (35, 11)}
        self.assertEqual(answer, sf.feature_constraint(("N", "M")))

    def test_find_isomorphism(self):
        sf = SentenceFilter(self.trees_testfile[1])
        sf.possible_token_pairs = {
            ("V", "S"): {(12, 11), (0, 4)},
            ("V", "N"): {(12, 23), (36, 39)},
            ("N", "BY"): {(39, 37)},
        }
        self.assertFalse(sf.find_isomorphism())
        sf.possible_token_pairs = {
            ("V", "S"): {(12, 11), (36, 35), (0, 4)},
            ("V", "N"): {(36, 39), (0, 3), (12, 23)},
            ("N", "BY"): {(39, 37)},
        }
        self.assertTrue(sf.find_isomorphism())

    def test_filter_sentence_false_constraints(self):
        # all tokens are found, but constraints are not observed
        sf = SentenceFilter(self.trees_testfile[4])
        self.assertFalse(
            sf.filter_sentence(by_passive[0], by_passive[1])
        )  # no ('V', 'N'): {'deprels': 'obl'}
        self.assertFalse(
            sf.filter_sentence(SOmatchingNumber[0], SOmatchingNumber[1])
        )  # no ('V', 'O'): {
        # 'deprels': 'obj'}
        self.assertTrue(sf.filter_sentence(ADPdistance[0], ADPdistance[1]))  # okay

    def test_filter_sentence_not_isomorphic(self):
        sf = SentenceFilter(self.trees_testfile[3])
        self.assertTrue(
            sf.filter_sentence(SOmatchingNumber[0], SOmatchingNumber[1])
        )  # okay
        sf = SentenceFilter(self.trees_testfile[6])
        self.assertFalse(
            sf.filter_sentence(SOmatchingNumber[0], SOmatchingNumber[1])
        )  # all all tokens are found, every constraint has a possible pair, but there is not isomorphism

    def test_filter_sentence_not_all_tokens(self):
        # some tokens are not found
        sf = SentenceFilter(self.trees_testfile[7])
        self.assertFalse(sf.filter_sentence(by_passive[0], by_passive[1]))
        self.assertFalse(sf.filter_sentence(ADPdistance[0], ADPdistance[1]))

    def test_filter_sentence_all_tokens_without_constraints(self):
        # find only tokens without constraints
        sf = SentenceFilter(self.trees_testfile[7])
        self.assertTrue(sf.filter_sentence({"be": {"lemma": "be"}}, {}))

    def test_filter_sentence_wrong_constraint_type(self):
        sf = SentenceFilter(self.trees_testfile[4])
        wrong_constraint_type_query = [
            {"P": {"upos": "ADP"}, "N": {"upos": "NOUN"}},
            {("N", "P"): {"wrong_constraint": "case"}},
        ]
        with self.assertRaises(KeyError):
            sf.filter_sentence(*wrong_constraint_type_query)
