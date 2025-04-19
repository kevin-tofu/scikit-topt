import os
from typing import Callable
from typing import Literal
import inspect
import shutil
import json
from dataclasses import dataclass, asdict
import numpy as np
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
    beta_step: int = 12
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
        if "eta_init" in cfg.__dataclass_fields__:
            self.schedulers.add(
                "eta",
                self.cfg.eta_init,
                self.cfg.eta,
                self.cfg.eta_step,
                self.cfg.max_iters
            )
        else:
            self.schedulers.add(
                "eta",
                self.cfg.eta,
                self.cfg.eta,
                -1,
                self.cfg.max_iters
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
        tsk = self.tsk
        cfg = self.cfg
        
        elements_volume_design = tsk.elements_volume[tsk.design_elements]
        elements_volume_design_sum = np.sum(elements_volume_design)

        rho = np.zeros_like(tsk.all_elements, dtype=np.float64)
        iter_begin = 1
        if cfg.restart is True:
            if cfg.restart_from > 0:
                data = np.load(
                    f"{cfg.dst_path}/data/{str(cfg.restart_from).zfill(6)}-rho.npz"
                )
            else:
                iter, data_path = misc.find_latest_iter_file(f"{cfg.dst_path}/data")
                data = np.load(data_path)
                iter_begin = iter

            rho[tsk.design_elements] = data["rho_design_elements"]
            del data
        else:
            # _vol_frac = cfg.vol_frac if cfg.vol_frac_step < 0 else cfg.vol_frac_init
            # rho += _vol_frac + 0.1 * (np.random.rand(len(tsk.all_elements)) - 0.5)
            # rho += _vol_frac + 0.15
            rho += 0.8 if cfg.vol_frac_step < 0 else cfg.vol_frac_init
            np.clip(rho, cfg.rho_min, cfg.rho_max, out=rho)

        if cfg.design_dirichlet is True:
            rho[tsk.force_elements] = 1.0
        else:
            rho[tsk.dirichlet_force_elements] = 1.0
        rho[tsk.fixed_elements_in_rho] = 1.0
        self.init_schedulers()
        
        if cfg.interpolation == "SIMP":
        # if False:
            # density_interpolation = composer.simp_interpolation_numba
            density_interpolation = composer.simp_interpolation
            dC_drho_func = derivatives.dC_drho_simp
        elif cfg.interpolation == "RAMP":
            density_interpolation = composer.ramp_interpolation
            dC_drho_func = derivatives.dC_drho_ramp
        else:
            raise ValueError("should be SIMP/RAMP")
        
        rho_prev = np.zeros_like(rho)
        rho_filtered = np.zeros_like(rho)
        rho_projected = np.zeros_like(rho)
        dH = np.empty_like(rho)
        grad_filtered = np.empty_like(rho)
        dC_drho_projected = np.empty_like(rho)
        strain_energy_ave = np.zeros_like(rho)
        compliance_avg = np.zeros_like(rho)
        dH = np.zeros_like(rho)

        # dC_drho_ave = np.zeros_like(rho)
        dC_drho_full = np.zeros_like(rho)
        dC_drho_ave = np.zeros_like(rho[tsk.design_elements])
        scaling_rate = np.empty_like(rho[tsk.design_elements])
        rho_candidate = np.empty_like(rho[tsk.design_elements])
        tmp_lower = np.empty_like(rho[tsk.design_elements])
        tmp_upper = np.empty_like(rho[tsk.design_elements])
        force_list = tsk.force if isinstance(tsk.force, list) else [tsk.force]
        filter_radius_prev = cfg.filter_radius_init if cfg.filter_radius_step > 0 else cfg.filter_radius
        self.helmholz_solver.update_radius(tsk.mesh, filter_radius_prev)
        for iter_loop, iter in enumerate(range(iter_begin, cfg.max_iters+iter_begin)):
            print(f"iterations: {iter} / {cfg.max_iters}")
            p, vol_frac, beta, move_limit, eta, percentile, filter_radius = (
                self.schedulers.values(iter)[k] for k in [
                    'p', 'vol_frac', 'beta', 'move_limit', 'eta', 'percentile', 'filter_radius'
                ]
            )
            if filter_radius_prev != filter_radius:
                print("Filter Update")
                self.helmholz_solver.update_radius(tsk.mesh, filter_radius)
            
            print(f"p {p:.4f}, vol_frac {vol_frac:.4f}, beta {beta:.4f}, move_limit {move_limit:.4f}")
            print(f"eta {eta:.4f}, percentile {percentile:.4f} filter_radius {filter_radius:.4f}")
            rho_prev[:] = rho[:]
            rho_filtered[:] = self.helmholz_solver.filter(rho)
            projection.heaviside_projection_inplace(
                rho_filtered, beta=beta, eta=cfg.beta_eta, out=rho_projected
            )
            dC_drho_ave[:] = 0.0
            dC_drho_full[:] = 0.0
            strain_energy_ave[:] = 0.0
            compliance_avg[:] = 0.0
            for force in force_list:
                dH[:] = 0.0
                compliance, u = solver.compute_compliance_basis(
                    tsk.basis, tsk.free_nodes, tsk.dirichlet_nodes, force,
                    tsk.E0, tsk.Emin, p, tsk.nu0,
                    rho_projected,
                    elem_func=density_interpolation
                )
                compliance_avg += compliance
                strain_energy = composer.strain_energy_skfem(
                    tsk.basis, rho_projected, u,
                    tsk.E0, tsk.Emin, p, tsk.nu0,
                    elem_func=density_interpolation
                )
                strain_energy_ave += strain_energy
                
                # rho_safe = np.clip(rho_filtered, 1e-3, 1.0)
                np.copyto(
                    dC_drho_projected,
                    dC_drho_func(
                        rho_projected,
                        strain_energy, tsk.E0, tsk.Emin, p
                    )
                )
                projection.heaviside_projection_derivative_inplace(
                    rho_filtered,
                    beta=beta, eta=cfg.beta_eta, out=dH
                )
                # dH[:] = projection.heaviside_projection_derivative(
                #     rho_filtered, beta=beta, eta=cfg.beta_eta
                # )
                np.multiply(dC_drho_projected, dH, out=grad_filtered)
                dC_drho_full[:] += self.helmholz_solver.gradient(grad_filtered)
                # dC_drho_ave[:] += dC_drho_full[tsk.design_elements]
                # dC_drho_dirichlet[:] += dC_drho_full[tsk.dirichlet_elements]
                
            dC_drho_full /= len(force_list)
            strain_energy_ave /= len(force_list)
            compliance_avg /= len(force_list)
            print(f"dC_drho_full- min:{dC_drho_full.min()} max:{dC_drho_full.max()}")
            
            filtered = self.helmholz_solver.filter(dC_drho_full)
            np.copyto(dC_drho_full, filtered)
            
            dC_drho_ave[:] = dC_drho_full[tsk.design_elements]
            rho_candidate[:] = rho[tsk.design_elements] # Dont forget. inplace
            
            # 
            self.rho_update(
                # iter_loop,
                iter,
                rho_candidate,
                rho_projected,
                dC_drho_ave,
                strain_energy_ave,
                scaling_rate,
                move_limit,
                eta,
                beta,
                tmp_lower,
                tmp_upper,
                percentile,
                density_interpolation,
                elements_volume_design,
                elements_volume_design_sum,
                vol_frac
            )
            # 
            rho[tsk.design_elements] = rho_candidate
            if cfg.design_dirichlet is True:
                rho[tsk.force_elements] = 1.0
            else:
                rho[tsk.dirichlet_force_elements] = 1.0

            filter_radius_prev = filter_radius
            # rho_diff = np.mean(np.abs(rho[tsk.design_elements] - rho_prev[tsk.design_elements]))
            print(
                f"scaling_rate min/mean/max {scaling_rate.min()} {scaling_rate.mean()} {scaling_rate.max()}"
            )
            self.recorder.feed_data("rho", rho[tsk.design_elements])
            self.recorder.feed_data("rho_projected", rho_projected[tsk.design_elements])
            self.recorder.feed_data("strain_energy", strain_energy_ave)
            self.recorder.feed_data("compliance", compliance_avg)
            self.recorder.feed_data("scaling_rate", scaling_rate)
            
            
            
            if iter % (cfg.max_iters // self.cfg.record_times) == 0 or iter == 1:
            # if True:
                print(f"Saving at iteration {iter}")
                self.recorder.print()
                # self.recorder_params.print()
                self.recorder.export_progress()
                
                visualization.save_info_on_mesh(
                    tsk,
                    rho_projected, rho_prev, strain_energy_ave,
                    cfg.vtu_path(iter),
                    cfg.image_path(iter, "rho"),
                    f"Iteration : {iter}",
                    cfg.image_path(iter, "strain_energy"),
                    f"Iteration : {iter}"
                )
                visualization.export_submesh(
                    tsk, rho, 0.5, f"{cfg.dst_path}/cubic_top.vtk"
                )
                np.savez_compressed(
                    f"{cfg.dst_path}/data/{str(iter).zfill(6)}-rho.npz",
                    rho_design_elements=rho[tsk.design_elements],
                    # compliance=compliance
                )

        visualization.rho_histo_plot(
            rho[tsk.design_elements],
            f"{self.cfg.dst_path}/mesh_rho/last.jpg"
        )
        visualization.export_submesh(
            tsk, rho, 0.5, f"{cfg.dst_path}/cubic_top.vtk"
        )


    def rho_update(
        self,
        iter_loop: int,
        rho_candidate: np.ndarray,
        rho_projected: np.ndarray,
        dC_drho_ave: np.ndarray,
        strain_energy_ave: np.ndarray,
        scaling_rate: np.ndarray,
        move_limit: float,
        eta: float,
        tmp_lower: np.ndarray,
        tmp_upper: np.ndarray,
        lambda_lower: float,
        lambda_upper: float,
        percentile: float,
        interploation: Callable,
        elements_volume_design: np.ndarray,
        elements_volume_design_sum: float,
        vol_frac: float
    ):
        raise NotImplementedError("")