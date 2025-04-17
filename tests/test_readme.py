import scitopt
import pytest


def oc_ramp_optimize(tsk):
    cfg = scitopt.core.optimizer.OC_Config()
    cfg.max_iters = 1
    cfg.record_times = 1

    optimizer = scitopt.core.OC_Optimizer(cfg, tsk)

    optimizer.parameterize()
    optimizer.optimize()


def moc_optimize(tsk):
    
    cfg = scitopt.core.optimizer.MOC_Config()
    cfg.max_iters = 1
    cfg.record_times = 1

    optimizer = scitopt.core.MOC_Optimizer(cfg, tsk)

    optimizer.parameterize()
    optimizer.optimize()


def kkt_optimize(tsk):
    
    cfg = scitopt.core.optimizer.KKT_Config()
    cfg.max_iters = 1
    cfg.record_times = 1

    optimizer = scitopt.core.KKT_Optimizer(cfg, tsk)

    optimizer.parameterize()
    optimizer.optimize()


def test_optimizers():
    tsk = scitopt.mesh.toy_problem.toy_test()
    oc_ramp_optimize(tsk)
    moc_optimize(tsk)
    kkt_optimize(tsk)