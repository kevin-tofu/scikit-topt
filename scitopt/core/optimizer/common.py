import os
from typing import Literal
import inspect
import shutil
import json
from dataclasses import dataclass, asdict
import numpy as np
import scipy
import scipy.sparse.linalg as spla
import skfem
import meshio
import scitopt
from scitopt import tools
from scitopt.core import derivatives, projection
from scitopt.core import visualization
from scitopt.fea import solver
from scitopt import filter
from scitopt.fea import composer
from scitopt.core import misc



@dataclass
class Sensitivity_Config():
    dst_path: str = "./result"
    interpolation: Literal["SIMP", "RAMP"] = "SIMP"
    record_times: int=20
    max_iters: int=200
    p_init: float = 1.0
    p: float = 3.0
    p_step: int = 4
    vol_frac_init: float = 0.8
    vol_frac: float = 0.4
    vol_frac_step: int = 3
    beta_init: float = 1.0
    beta: float = 3
    beta_step: float = 12.
    beta_eta: float = 0.50
    eta: float = 0.5
    percentile_init: float = 60
    percentile: float = 90
    percentile_step: int = 3
    filter_radius_init: float = 0.2
    filter_radius: float = 0.05
    filter_radius_step: int = 3
    mu_p: float = 2.0
    rho_min: float = 1e-3
    rho_max: float = 1.0
    move_limit_init: float = 0.3
    move_limit: float = 0.14
    move_limit_step: int = 3
    restart: bool = False
    restart_from: int = -1
    export_img: bool = False
    design_dirichlet: bool=False
    lambda_lower: float=1e-2
    lambda_upper: float=1e+2


    @classmethod
    def from_defaults(cls, **args):
        sig = inspect.signature(cls)
        valid_keys = sig.parameters.keys()
        filtered_args = {k: v for k, v in args.items() if k in valid_keys}
        return cls(**filtered_args)

    
    def export(self, path: str):
        with open(f"{path}/cfg.json", "w") as f:
            json.dump(asdict(self), f, indent=2)

    def vtu_path(self, iter: int):
        return f"{self.dst_path}/mesh_rho/info_mesh-{iter:08d}.vtu"


    def image_path(self, iter: int, prefix: str):
        if self.export_img:
            return f"{self.dst_path}/mesh_rho/info_{prefix}-{iter:08d}.jpg"
        else:
            return None



class Sensitivity_Analysis():
    def __init__(
        self,
        cfg: Sensitivity_Config,
        tsk: scitopt.mesh.TaskConfig,
    ):
        self.cfg = cfg
        self.tsk = tsk
        if not os.path.exists(self.cfg.dst_path):
            os.makedirs(self.cfg.dst_path)
        self.cfg.export(self.cfg.dst_path)
        # self.tsk.nodes_and_elements_stats(self.cfg.dst_path)
        
        if cfg.design_dirichlet is False:
            self.tsk.exlude_dirichlet_from_design()
        
        if cfg.restart is True:
            self.load_parameters()
        else:
            if os.path.exists(f"{self.cfg.dst_path}/mesh_rho"):
                shutil.rmtree(f"{self.cfg.dst_path}/mesh_rho")
            os.makedirs(f"{self.cfg.dst_path}/mesh_rho")
            if not os.path.exists(f"{self.cfg.dst_path}/data"):
                os.makedirs(f"{self.cfg.dst_path}/data")
            
            self.parameterize()

        self.recorder = tools.HistoriesLogger(self.cfg.dst_path)
        self.recorder.add("rho", plot_type="min-max-mean-std")
        self.recorder.add("rho_projected", plot_type="min-max-mean-std")
        self.recorder.add("strain_energy", plot_type="min-max-mean-std")
        self.recorder.add("vol_error")
        self.recorder.add("compliance", ylog=True)
        self.recorder.add("scaling_rate", plot_type="min-max-mean-std")
        # self.recorder.add("dC", plot_type="min-max-mean-std")
        # self.recorder.add("lambda_v", ylog=False) # True
        self.schedulers = tools.Schedulers(self.cfg.dst_path)
    
    
    def init_schedulers(self, export: bool=True):

        cfg = self.cfg
        p_init = cfg.p_init
        vol_frac_init = cfg.vol_frac_init
        move_limit_init = cfg.move_limit_init
        beta_init = cfg.beta_init
        self.schedulers.add(
            "p",
            p_init,
            cfg.p,
            cfg.p_step,
            cfg.max_iters
        )
        self.schedulers.add(
            "vol_frac",
            vol_frac_init,
            cfg.vol_frac,
            cfg.vol_frac_step,
            cfg.max_iters
        )
        # print(move_init)
        # print(cfg.move_limit, cfg.move_limit_step)
        self.schedulers.add(
            "move_limit",
            move_limit_init,
            cfg.move_limit,
            cfg.move_limit_step,
            cfg.max_iters
        )
        self.schedulers.add(
            "beta",
            beta_init,
            cfg.beta,
            cfg.beta_step,
            cfg.max_iters
        )
        self.schedulers.add(
            "percentile",
            cfg.percentile_init,
            cfg.percentile,
            cfg.percentile_step,
            cfg.max_iters
        )
        self.schedulers.add(
            "filter_radius",
            cfg.filter_radius_init,
            cfg.filter_radius,
            cfg.filter_radius_step,
            cfg.max_iters
        )
        if export:
            self.schedulers.export()


    def parameterize(self):
        self.helmholz_solver = filter.HelmholtzFilter.from_defaults(
            self.tsk.mesh,
            self.cfg.filter_radius,
            solver_type="pyamg",
            dst_path=f"{self.cfg.dst_path}/data",
            
        )

    def load_parameters(self):
        self.helmholz_solver = filter.HelmholtzFilter.from_file(
            f"{self.cfg.dst_path}/data"
        )
    
    def optimize(self):
        raise NotImplementedError("")

