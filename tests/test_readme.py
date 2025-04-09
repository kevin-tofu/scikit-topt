import scitopt
import pytest


def oc_ramp_optimize(tsk):
    cfg = scitopt.core.OC_Config()
    cfg.max_iters = 1
    cfg.record_times = 1

    optimizer = scitopt.core.OC_Optimizer(cfg, tsk)

    optimizer.parameterize(preprocess=True)
    optimizer.optimize()


def moc_ramp_optimize(tsk):
    
    cfg = scitopt.core.MOC_Config()
    cfg.max_iters = 1
    cfg.record_times = 1

    optimizer = scitopt.core.MOC_Optimizer(cfg, tsk)

    optimizer.parameterize(preprocess=True)
    optimizer.optimize()


def test_main():
    tsk = scitopt.mesh.toy_problem.toy_test()
    oc_ramp_optimize(tsk)
    moc_ramp_optimize(tsk)