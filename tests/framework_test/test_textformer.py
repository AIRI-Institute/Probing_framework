import unittest
from pathlib import Path

import numpy as np
import pytest

from probing.config import data_folder
from probing.data_former import TextFormer


def sort_dict(dictionary):
    dictionary = {k: np.sort(v, axis=0) for k, v in dictionary.items()}
    return dictionary

@pytest.mark.data_former
class TestTextFormer(unittest.TestCase):
    path_testfile = Path(Path(__file__).parent.resolve(), "test_sent_len.txt")
    unique_labels_reference = {'0', '1', '2'}
    samples_reference = {'tr': np.array([('Район традиционно развивается как сельскохозяйственный .', '0'),
                                         ('Он посвящен лицам года в американской политике .', '1'),
                                         ('Ведомства проверяют торговые операции банка по покупке и продаже валюты .', '2'),
                                         ]),
                         'te': np.array([('Ангел стал известен как опытный духовник .', '0'),
                                         ('Нити в верхней части жидкости — ветер .', '1'),
                                         ('Я стала другой : терпимой , спокойной и дружелюбной ко всем .', '2')
                                         ])
                         }

    def test_data_path(self):
        task_name = 'sent_len'
        data = TextFormer(
                probe_task=task_name
            )
        task_path = Path(data.data_path)
        self.assertEqual(task_path.parent, data_folder)
        self.assertEqual(task_path, Path(data_folder, f'{task_name}.txt'))

    def test_unique_labels(self):
        task_name = 'test_sent_len'
        data = TextFormer(
            probe_task=task_name,
            data_path=self.path_testfile,
        )
        samples, unique_labels = data.form_data()
        self.assertEqual(unique_labels, self.unique_labels_reference)

    def test_samples_dict(self):
        task_name = 'test_sent_len'
        data = TextFormer(
            probe_task=task_name,
            data_path=self.path_testfile
        )
        samples, unique_labels = data.form_data()

        samples = sort_dict(samples)
        self.samples_reference = sort_dict(self.samples_reference)

        samples_keys = set(samples.keys())
        self.samples_ref_keys = set(self.samples_reference.keys())
        self.assertEqual(samples_keys, self.samples_ref_keys)

        for k in samples_keys:
            np.testing.assert_equal(samples[k], self.samples_reference[k])
