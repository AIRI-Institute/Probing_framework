import unittest
from pathlib import Path

import pytest

from probing.data_former import TextFormer
from probing.pipeline import ProbingPipeline


@pytest.mark.truncation
class TestTruncation(unittest.TestCase):
    path_testfile = Path(Path(__file__).parent.resolve(), "test_cs_cac_AdpType.csv")
    task_data = TextFormer(Path(path_testfile).stem, path_testfile)
    task_dataset, num_classes = task_data.samples, len(task_data.unique_labels)

    experiment1 = ProbingPipeline(
        hf_model_name="bert-base-multilingual-uncased", device="cpu"
    )
    experiment2 = ProbingPipeline(hf_model_name="t5-small", device="cpu")

    def test_launch_right(self):
        tokenized_text = self.experiment1.transformer_model.tokenize_text(
            list(self.task_dataset["tr"][:, 0])
        )
        _, _, excluded_rows = self.experiment1.transformer_model._fix_tokenized_tensors(
            tokenized_text
        )
        tokenized_tensor = tokenized_text["input_ids"][
            :, self.experiment1.transformer_model.tokenizer.model_max_length :
        ]
        gold_answer = tokenized_tensor[tokenized_tensor[:, 0] != 0].shape[0]
        self.assertEqual(len(excluded_rows), gold_answer)

    def test_launch_left(self):
        tokenized_text = self.experiment2.transformer_model.tokenize_text(
            list(self.task_dataset["tr"][:, 0])
        )
        _, _, excluded_rows = self.experiment2.transformer_model._fix_tokenized_tensors(
            tokenized_text
        )
        tokenized_tensor = tokenized_text["input_ids"][
            :, self.experiment2.transformer_model.tokenizer.model_max_length :
        ]
        gold_answer = tokenized_tensor[tokenized_tensor[:, 0] != 0].shape[0]
        self.assertEqual(len(excluded_rows), gold_answer)
