import argparse
import unittest
from .execution import check_correctness
from .models import get_model
from rkgb import Result
from rockmate import Rockmate
import torch

class RockmateTest(unittest.TestCase):
    MODEL = "resnet18"
    def test_valid(self):
        
        model, sample = get_model(self.MODEL,  device="cuda", batchsize=1)
        out_g, out_c, out_r = check_correctness(model, sample)
        rtol = torch.abs((out_g - out_c.to("cuda"))/(out_g)).mean()
        self.assertTrue(torch.allclose(out_r[0], out_g[0], rtol=float(rtol)))
