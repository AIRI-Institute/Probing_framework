from typing import List, Dict
from enum import Enum


# categories from UD annotation
ud_categories: List[Enum] = [
    "PronType", "Gender", "VerbForm", "NumType",
    "Animacy", "Mood", "Poss", "NounClass", "Tense",
    "Reflex", "Number", "Aspect", "Foreign", "Case",
    "Voice", "Abbr", "Definite", "Evident", "Typo",
    "Degree", "Polarity", "Person", "Polite", "Clusivity"
]

partitions_by_files: Dict[str, List[List[float]]] = {
    "one_file": [[0.8], [0.1], [0.1]],
    "two_files": [[1.0], [0.5], [0.5]],
    "three_files": [[1.0], [1.0], [1.0]]
    }
