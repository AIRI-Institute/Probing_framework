import unittest
from pathlib import Path

import pytest

from probing.pipeline import ProbingPipeline


@pytest.mark.pipeline
class TestTextFormer(unittest.TestCase):
    path_testfile = Path(Path(__file__).parent.resolve(), "test_gapping.txt")
    experiment = ProbingPipeline(
        hf_model_name="bert-base-multilingual-uncased",
        device="cpu",
        classifier_name="logreg",
    )

    def test_launch1(self):
        task_name = "test_gapping"
        self.experiment.run(
            probe_task=task_name,
            path_to_task_file=self.path_testfile,
            verbose=True,
            train_epochs=2,
        )
        return True

    def test_launch2(self):
        task_name = "test_gapping"
        self.experiment.run(
            probe_task=task_name,
            path_to_task_file=self.path_testfile,
            verbose=True,
            train_epochs=2,
            do_control_task=True,
        )
        return True
