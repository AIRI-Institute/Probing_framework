import unittest
import torch
from pathlib import Path

import pytest

from probing.classifier import LinearVariational, MDLLinearModel
from probing.utils import KL


@pytest.mark.classifier

class TestClassifier(unittest.TestCase):
    input_dim = 512
    num_classes = 2
    
    def test_MDLLinearModel(self):
        model = MDLLinearModel(input_dim=self.input_dim,
                               num_classes=self.num_classes)
        x = torch.rand(3, self.input_dim)
       
        preds = model.forward(x)
        self.assertEqual(list(preds.shape), [x.size()[0], self.num_classes])
    
    def test_LinearVariational(self):
        model = LinearVariational(in_features=self.input_dim,
                                  out_features=self.num_classes,
                                  parent=KL)
        
        mu = torch.nn.Parameter(
            torch.FloatTensor(self.input_dim, self.num_classes).normal_(mean=0, std=0.001)
        )
        p = torch.nn.Parameter(
            torch.FloatTensor(self.input_dim, self.num_classes).normal_(mean=0, std=0.001)
        )
        parameter = model._reparameterize(mu, p)
        self.assertEqual(type(parameter), torch.Tensor)
        
        x = torch.rand(self.input_dim)
        preds = model.forward(x)
        self.assertEqual(list(preds.shape), [self.num_classes])
        self.assertEqual(type(model.parent.accumulated_kl_div), float)
        