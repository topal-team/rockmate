import argparse
import unittest
from .execution import check_correctness, exec_pt, exec_rk, execution
from .models import get_model
from rkgb import Result
from rockmate import Rockmate
from rockmate.solvers import CheapSolver
import torch

class RockmateTest(unittest.TestCase):
    MODEL = "resnet18"
    def test_valid(self):
        model, sample = get_model(self.MODEL,  device="cuda", batchsize=1)
        out_g, out_c, out_r = check_correctness(model, sample)
        rtol = torch.abs((out_g - out_c.to("cuda"))/(out_g)).mean()
        self.assertTrue(torch.allclose(out_r[0], out_g[0], rtol=float(rtol)))

    def test_mem(self):
        model, sample = get_model(self.MODEL,  device="cuda", batchsize=16)
        for p in model.parameters():
            p.grad = torch.ones_like(p)
        cheap_solver = CheapSolver(add_offload=False)
        rkmod = Rockmate(model, sample, 
                         budget=torch.cuda.get_device_properties(0).total_memory,
                         solve_sched=False)
        rkmod.preprocess()
        rkmod.op_sched = cheap_solver(rkmod.rkgb_res.hierarchical_cluster)[0]
        rkmod.get_compiled_fct()
        simulate_mem = rkmod.op_sched.peak_mem
        torch.cuda.reset_peak_memory_stats()
        mem = torch.cuda.memory_allocated()
        _ = execution(rkmod, sample, niters=5, zero_grad=False)
        peak_mem = torch.cuda.max_memory_allocated() - mem
        self.assertAlmostEqual(peak_mem/peak_mem, simulate_mem/peak_mem, places=3)
