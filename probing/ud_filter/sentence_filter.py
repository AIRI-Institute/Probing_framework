import re
from conllu import models
from itertools import product
from collections import defaultdict
from typing import Iterable
import copy


class SentenceFilter:
    def __init__(self, sentence: models.TokenList):
        self.sentence = sentence
        self.node_pattern = None
        self.constraints = None
        self.sent_deprels = {}
        self.nodes_tokens = None
        self.possible_pairs = None

    def token_match_node(self, token: models.Token, node_pattern: dict) -> bool:
        """Checks if token matches node_pattern"""

        for feat in node_pattern:
            if feat in token.keys():
                if not re.fullmatch(node_pattern[feat], token[feat], re.I):  # changed from re.match to re.fullmatch
                    return False
            elif token['feats']:
                if feat in token['feats']:
                    if not re.fullmatch(node_pattern[feat], token['feats'][feat],
                                        re.I):  # changed from re.match to re.fullmatch
                        return False
                elif feat == "exclude":
                    for ef in node_pattern[feat]:
                        if ef in token['feats']:
                            return False
                else:
                    return False
            else:
                return False
        return True

    def all_deprels(self, token_list: models.TokenList) -> defaultdict:
        """Creates dictionary {'relation': [(head, dependent)]} of all relations in the sentence"""

        deprels = defaultdict(list)
        for t in token_list:
            if isinstance(t['head'], int) and isinstance(t['id'], int):
                deprels[t['deprel']].append((t['head'] - 1, t['id'] - 1))
        return deprels

    def _search_suitable_tokens(self, node: str):
        """Selects from token_list those tokens that match given node_pattern"""

        for token in self.sentence:
            if self.token_match_node(token, self.node_pattern[node]) and isinstance(token['id'], int):
                self.nodes_tokens[node].append(token['id'] - 1)

    def _find_all_nodes(self):
        """For every node in pattern searches for matching tokens"""

        for node in self.node_pattern:
            self._search_suitable_tokens(node)
            if not self.nodes_tokens[node]:
                return False
        return True

    def _pattern_relations(self, rel_pattern: str):
        """Finds all relation names in the sentence that match given pattern"""

        rels = []
        for rel in self.sent_deprels:
            if re.fullmatch(rel_pattern, rel, re.I):  # changed from re.serach to re.fullmatch
                rels.append(rel)
        return rels

    def _pairs_with_rel(self, node_pair: tuple, rel_name: str) -> set:
        """Pairs of tokens with rel_name relation among possible pairs"""

        if rel_name not in self.sent_deprels:
            return set()
        else:
            return set(self.sent_deprels[rel_name]).intersection(self.possible_pairs[node_pair])

    def _pairs_matching_relpattern(self, node_pair: tuple) -> set:
        all_suitable_rels = set()
        for rel in self._pattern_relations(self.constraints[node_pair]['deprels']):
            all_suitable_rels = all_suitable_rels | self._pairs_with_rel(node_pair, rel)
        return all_suitable_rels

    def _linear_distance(self, node_pair: tuple) -> set:
        """Searches pairs with given linear distance between tokens
        :param lindist: tuple(min_distance, max_distance)"""

        suitable_pairs = set()
        lindist = self.constraints[node_pair]['lindist']
        for pair in self.possible_pairs[node_pair]:
            dist = pair[1] - pair[0]
            if lindist[0] <= dist <= lindist[1]:
                suitable_pairs.add(pair)
        return suitable_pairs

    def _pair_match_fconstraint(self, token_pair: tuple, fconstraint: dict) -> bool:
        """Checks if token pair matches all constraint on features"""

        t1_feats = self.sentence[token_pair[0]]['feats']
        t2_feats = self.sentence[token_pair[1]]['feats']
        if t1_feats and t2_feats:
            for ctype in fconstraint:
                for f in fconstraint[ctype]:
                    if (f in t1_feats) and (f in t2_feats):
                        if ctype == 'intersec':
                            if t1_feats[f] != t2_feats[f]:
                                return False
                        elif ctype == 'disjoint':
                            if t1_feats[f] == t2_feats[f]:
                                return False
                        else:
                            raise ValueError('Wrong feature constraint type')
                    else:
                        return False
            return True
        else:
            return False

    def _feature_constraint(self, node_pair: tuple) -> set:
        """Finds all pairs that match constraint on features"""

        suitable_pairs = set()
        fconstraint = self.constraints[node_pair]['fconstraint']
        for pair in self.possible_pairs[node_pair]:
            if self._pair_match_fconstraint(pair, fconstraint):
                suitable_pairs.add(pair)
        return suitable_pairs

    def _match_constraints(self):
        """Checks if there is a token pair that matches all constraints"""
        prev = copy.copy(self.nodes_tokens)
        for np in self.constraints:
            self.possible_pairs[np] = list(product(self.nodes_tokens[np[0]], self.nodes_tokens[np[1]]))
            for constraint in self.constraints[np]:
                if constraint == 'deprels':
                    self.possible_pairs[np] = self._pairs_matching_relpattern(np)
                elif constraint == 'lindist':
                    self.possible_pairs[np] = self._linear_distance(np)
                elif constraint == 'fconstraint':
                    self.possible_pairs[np] = self._feature_constraint(np)
                else:
                    raise ValueError('Wrong constraint type')
                if not self.possible_pairs[np]:
                    return False
                else:
                    self.nodes_tokens[np[0]] = set([p[0] for p in self.possible_pairs[np]])
                    self.nodes_tokens[np[1]] = set([p[1] for p in self.possible_pairs[np]])
        if prev != self.nodes_tokens:
            self._match_constraints()
        return True

    def filter_sentence(self, node_pattern: dict, constraints: dict):
        self.node_pattern = node_pattern
        self.constraints = constraints
        self.nodes_tokens = {node: [] for node in self.node_pattern}
        self.possible_pairs = {pair: set() for pair in self.constraints}
        if not self._find_all_nodes():
            return False
        else:
            self.sent_deprels = self.all_deprels(self.sentence)
            if self._match_constraints():
                return True
            else:
                return False
