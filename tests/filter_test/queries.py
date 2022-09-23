by_passive = [
    {
        'V': {},
        'S': {},
        'BY': {'lemma': 'by'},
        'N': {},
    },
    {
        ('V', 'S'): {'deprels': 'aux:pass'},
        ('V', 'N'): {'deprels': 'obl'},
        ('N', 'BY'): {'deprels': 'case'},
    }
]

SOmatchingNumber = [
    {
        'S': {},
        'V': {},
        'O': {}
    },
    {
        ('V', 'S'): {'deprels': '.subj'},
        ('V', 'O'): {'deprels': 'obj'},
        ('S', 'O'): {'fconstraint': {'intersec': ['Number']}}
    }
]

ADPdistance = [
    {
        'P': {'upos': 'ADP'},
        'N': {}
    },
    {
        ('N', 'P'): {
            'deprels': 'case',
            'lindist': (-6, -4)}
    }
]