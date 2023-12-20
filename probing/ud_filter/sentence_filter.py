import re
from collections import defaultdict
from itertools import product
from typing import Any, DefaultDict, Dict, List, Set, Tuple

import networkx as nx
from conllu import models

from probing.ud_filter.utils import check_query


class SentenceFilter:
    """
    Checks if a sentence matches patterns

    Attributes:
        sentence: sentence represented as TokenList
        node_pattern: a dictionary with a node_label as a key and a dictionary of feature restrictions as a value.
        sample_node_pattern = { node_label: {
                                    field_or_category: regex_pattern,
                                    'exclude': [exclude categories]} }
        constraints: a dictionary with a node_pair as a key and a dictionary with different constraints on node_pair
        sample_constraints = { ('W1', 'W2'): {
                                'deprels': regexp_pattern (W1 as head, W2 as dependent),
                                'fconstraint': {
                                    'disjoint': [grammar_category],
                                    'intersec': [grammar_category]},
                                'lindist': (start, end) (relatively W1)} }
        sent_deprels: a dictionary of all relations and pairs with these relations {relation: [(head, dependent)]}
        nodes_tokens: a dictionary with all tokens that can be a node in the pattern {node: [token id]},
                    if filter_sentence == True, saves only one instance in nodes_tokens
        possible_token_pairs: a dictionary with all nodes pairs as a key and a list of possible token,
        pairs as a value, if filter_sentence == True, saves only one instance in possible_token_pairs
    """

    def __init__(self, sentence: models.TokenList):
        self.sentence = sentence
        self.node_pattern: Dict[str, Dict[str, str]] = {}
        self.constraints: Dict[Tuple[str, str], Dict[Any, Any]] = {}
        self.sent_deprels: DefaultDict[str, List[Tuple[int, int]]] = defaultdict(list)
        self.nodes_tokens: Dict[str, List[int]] = {}
        self.possible_token_pairs: Dict[Tuple[str, str], Set[Tuple[int, int]]] = {}

    def token_match_node(
        self, token: models.Token, node_pattern: Dict[str, str]
    ) -> bool:
        """Checks if a token matches the node_pattern"""

        for feat in node_pattern:
            if feat in token.keys():
                if not re.fullmatch(node_pattern[feat], token[feat], re.I):
                    return False
            elif token["feats"]:
                if feat in token["feats"]:
                    if not re.fullmatch(node_pattern[feat], token["feats"][feat], re.I):
                        return False
                elif feat == "exclude":
                    for ef in node_pattern[feat]:
                        if ef in token["feats"]:
                            return False
                else:
                    return False
            else:
                return False
        return True

    def all_deprels(self) -> DefaultDict[str, List[Tuple[int, int]]]:
        """Returns a dictionary {relation: [(head, dependent)]} of all relations in the sentence"""

        deprels: DefaultDict[str, list] = defaultdict(list)
        for token in self.sentence:
            if isinstance(token["head"], int) and isinstance(token["id"], int):
                deprels[token["deprel"]].append((token["head"] - 1, token["id"] - 1))
        return deprels

    def search_suitable_tokens(self, node: str) -> None:
        """Selects from a token_list those tokens that match given node_pattern
        and saves it in self.nodes_tokens[node]"""

        for token in self.sentence:
            if self.token_match_node(token, self.node_pattern[node]) and isinstance(
                token["id"], int
            ):
                self.nodes_tokens[node].append(token["id"] - 1)

    def find_all_nodes(self) -> bool:
        """Checks if every node in the pattern has at least one matching token"""

        for node in self.node_pattern:
            self.search_suitable_tokens(node)
            if not self.nodes_tokens[node]:
                return False
        return True

    def pattern_relations(self, rel_pattern: str) -> List[str]:
        """Returns all relation names in the sentence that match the given pattern"""

        rels = []
        for rel in self.sent_deprels:
            if re.fullmatch(
                rel_pattern, rel, re.I
            ):  # changed from re.serach to re.fullmatch
                rels.append(rel)
        return rels

    def pairs_with_rel(
        self, node_pair: Tuple[str, str], rel_name: str
    ) -> Set[Tuple[int, int]]:
        """Returns those pairs of tokens that:
        1) are related by a rel_name
        2) are among possible_token_pairs for a node_pair"""

        if rel_name not in self.sent_deprels:
            return set()
        else:
            return set(self.sent_deprels[rel_name]).intersection(
                self.possible_token_pairs[node_pair]
            )

    def pairs_matching_relpattern(
        self, node_pair: Tuple[str, str]
    ) -> Set[Tuple[int, int]]:
        """Returns a set of token pairs, whose relations match the pattern"""

        all_suitable_rels: Set[Tuple[int, int]] = set()
        for rel in self.pattern_relations(self.constraints[node_pair]["deprels"]):
            all_suitable_rels = all_suitable_rels | self.pairs_with_rel(node_pair, rel)
        return all_suitable_rels

    def linear_distance(self, node_pair: Tuple[str, str]) -> Set[Tuple[int, int]]:
        """Returns a set of token pairs with a given linear distance between tokens"""

        suitable_pairs: Set[Tuple[int, int]] = set()
        lindist = self.constraints[node_pair]["lindist"]
        for pair in self.possible_token_pairs[node_pair]:
            dist = pair[1] - pair[0]
            if lindist[0] <= dist <= lindist[1]:
                suitable_pairs.add(pair)
        return suitable_pairs

    def pair_match_fconstraint(
        self, token_pair: Tuple[int, int], fconstraint: Dict[Any, Any]
    ) -> bool:
        """Checks if a token pair matches all the feature constraints"""
        t1_feats = self.sentence[token_pair[0]]["feats"]
        t2_feats = self.sentence[token_pair[1]]["feats"]
        if t1_feats and t2_feats:
            for ctype in fconstraint:
                for f in fconstraint[ctype]:
                    if (f in t1_feats) and (f in t2_feats):
                        if ctype == "intersec":
                            if t1_feats[f] != t2_feats[f]:
                                return False
                        elif ctype == "disjoint":
                            if t1_feats[f] == t2_feats[f]:
                                return False
                        else:
                            raise ValueError("Wrong feature constraint type")
                    else:
                        return False
            return True
        else:
            return False

    def feature_constraint(self, node_pair: Tuple[str, str]) -> Set[Tuple[int, int]]:
        """Returns all pairs that match constraints on features"""

        suitable_pairs: Set[Tuple[int, int]] = set()
        fconstraint = self.constraints[node_pair]["fconstraint"]
        for pair in self.possible_token_pairs[node_pair]:
            if self.pair_match_fconstraint(pair, fconstraint):
                suitable_pairs.add(pair)
        return suitable_pairs

    def find_isomorphism(self) -> bool:
        """Checks if there is at least one graph with possible_token_pairs
        that is isomorphic to a constraint pairs graph"""

        nodes_graph = nx.Graph()
        nodes_graph.add_edges_from(list(self.possible_token_pairs.keys()))
        possible_edges = list(product(*self.possible_token_pairs.values()))
        for edges in possible_edges:
            tokens_graph = nx.Graph()
            tokens_graph.add_edges_from(edges)
            if len(tokens_graph.nodes) != len(nodes_graph.nodes):
                continue
            if nx.is_isomorphic(tokens_graph, nodes_graph):
                self.possible_token_pairs = {
                    k: {edges[i]} for i, k in enumerate(self.possible_token_pairs)
                }
                self.nodes_tokens = {
                    np[i]: list(self.possible_token_pairs[np])[0][i]
                    for np in self.possible_token_pairs
                    for i in range(2)
                }
                return True
        return False

    def match_constraints(self) -> bool:
        """Checks if there is at least one token pair that matches all constraints."""

        for np in self.constraints:
            self.possible_token_pairs[np] = set(
                product(self.nodes_tokens[np[0]], self.nodes_tokens[np[1]])
            )
            for constraint in self.constraints[np]:
                if constraint == "deprels":
                    self.possible_token_pairs[np] = self.pairs_matching_relpattern(np)
                elif constraint == "lindist":
                    self.possible_token_pairs[np] = self.linear_distance(np)
                elif constraint == "fconstraint":
                    self.possible_token_pairs[np] = self.feature_constraint(np)
                # else:
                #     raise ValueError("Wrong constraint type")
                # (not possible, this is controlled by ud_filter.utils.check_constraints)
                if not self.possible_token_pairs[np]:
                    return False
                else:
                    self.nodes_tokens[np[0]] = list(
                        set([p[0] for p in self.possible_token_pairs[np]])
                    )
                    self.nodes_tokens[np[1]] = list(
                        set([p[1] for p in self.possible_token_pairs[np]])
                    )
        if not self.find_isomorphism():
            return False
        return True

    def filter_sentence(
        self,
        node_pattern: Dict[str, Dict[str, str]],
        constraints: Dict[Tuple[str, str], dict],
    ) -> bool:
        """Check if a sentence contains at least one instance of a node_pattern that matches
        all the given and isomophism constraints"""
        check_query(node_pattern, constraints)
        self.node_pattern = node_pattern
        self.constraints = constraints
        self.nodes_tokens = {node: [] for node in self.node_pattern}
        self.possible_token_pairs = {pair: set() for pair in self.constraints}
        if not self.find_all_nodes():
            return False

        self.sent_deprels = self.all_deprels()
        if self.match_constraints():
            return tuple(self.nodes_tokens.values())

        return False
