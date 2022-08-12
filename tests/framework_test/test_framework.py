import pytest
import unittest
from pathlib import Path

from probing.data_former import TextFormer, EncodeLoader
from probing.config import data_folder


@pytest.mark.data_former
class TestTextFormer(unittest.TestCase):

    def test1(self):
        task_name = 'sent_len'
        data = TextFormer(
                probe_task=task_name
            )
        task_path = Path(data.data_path)
        self.assertEqual(task_path.parent, data_folder)
        self.assertEqual(task_path, Path(data_folder, f'{task_name}.txt'))
