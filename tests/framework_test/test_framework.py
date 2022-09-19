import pytest
import unittest
from pathlib import Path

from probing.data_former import TextFormer
from probing.config import data_folder


@pytest.mark.data_former
class TestTextFormer(unittest.TestCase):
    path_testfile = Path(Path(__file__).parent.resolve(), "test_sent_len.txt")
    unique_labels_reference = {'0', '1', '2'}
    samples_reference = {'tr': [('Район традиционно развивается как сельскохозяйственный .', '0'),
                                ('Он посвящен лицам года в американской политике .', '1'),
                                ('Ведомства проверяют торговые операции банка по покупке и продаже валюты .', '2')
                                ],
                         'te': [('Ангел стал известен как опытный духовник .', '0'),
                                ('Нити в верхней части жидкости — ветер .', '1'),
                                ('Я стала другой : терпимой , спокойной и дружелюбной ко всем .', '2')
                                ]
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

    def test_samples_unshuffle(self):
        task_name = 'test_sent_len'
        data = TextFormer(
            probe_task=task_name,
            data_path=self.path_testfile,
            shuffle=False
        )

        samples, unique_labels = data.form_data()
        self.assertDictEqual(samples, self.samples_reference)

    def test_samples_shuffle(self):
        task_name = 'test_sent_len'
        data = TextFormer(
            probe_task=task_name,
            data_path=self.path_testfile,
            shuffle=True
        )

        samples, unique_labels = data.form_data()
        for k, v in samples.items():
            self.assertEqual(type(v), list)
            self.assertEqual(type(v[0]), tuple)
        self.assertNotEqual(samples, self.samples_reference)
