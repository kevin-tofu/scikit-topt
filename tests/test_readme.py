import scitopt

tsk = scitopt.mesh.toy_problem.toy()
cfg = scitopt.core.OC_RAMP_Config()
cfg.max_iters = 1
cfg.record_times = 1

optimizer = scitopt.core.OC_Optimizer(cfg, tsk)

optimizer.parameterize(preprocess=True)
optimizer.optimize()
