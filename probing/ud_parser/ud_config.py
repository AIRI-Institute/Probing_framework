from typing import List
from enum import Enum


#  categories from UD annotation
ud_categories: List[Enum] = [
    "PronType", "Gender", "VerbForm", "NumType",
    "Animacy", "Mood", "Poss", "NounClass", "Tense",
    "Reflex", "Number", "Aspect", "Foreign", "Case",
    "Voice", "Abbr", "Definite", "Evident", "Typo",
    "Degree", "Polarity", "Person", "Polite", "Clusivity"
]