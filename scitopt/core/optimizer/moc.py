import os
import math
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
class MOC_RAMP_Config():
    dst_path: str = "./result"
    record_times: int=20
    max_iters: int=200
    p_init: float = 1.0
    p: float = 3.0
    p_rate: float = 20.0
    vol_frac_init: float = 0.8
    vol_frac: float = 0.4
    vol_frac_rate: float = 20.0
    beta_init: float = 1.0
    beta: float = 16
    beta_rate: float = 20.
    beta_eta: float = 0.5
    filter_radius: float = 0.5
    eta: float = 0.3
    mu_p: float = 100.0
    mu_d: float = 200.0
    mu_i: float = 10.0
    lambda_v: float = 1.0
    lambda_decay: float = 0.95
    rho_min: float = 1e-3
    rho_max: float = 1.0
    move_limit_init: float = 0.8
    move_limit: float = 0.2
    move_limit_rate: float = 20.0
    restart: bool = False
    restart_from: int = -1
    
    @classmethod
    def from_defaults(cls, **args):
        sig = inspect.signature(cls)
        valid_keys = sig.parameters.keys()
        filtered_args = {k: v for k, v in args.items() if k in valid_keys}
        return cls(**filtered_args)

    
    def export(self, path: str):
        with open(f"{path}/cfg.json", "w") as f:
            json.dump(asdict(self), f, indent=2)


def oc_update_with_projection(rho, dC, move, eta, rho_min, rho_max):
    
    dC_c = dC - np.mean(dC)
    scale = np.sign(-dC_c) * (np.abs(dC_c) / (np.mean(np.abs(dC_c)) + 1e-8)) ** eta
    rho_new = rho * scale
    rho_new = np.clip(rho_new, rho - move, rho + move)
    rho_new = np.clip(rho_new, rho_min, rho_max)
    return rho_new


def oc_log_update(rho, dC, move, eta, rho_min, rho_max):
    eps = 1e-8
    # dC_centered = dC - np.mean(dC)

    scaling = -dC / (np.mean(np.abs(dC)) + eps)
    scaling = np.clip(scaling, 1e-3, 1e3)
    log_rho = np.log(np.clip(rho, rho_min, 1.0))

    rho_new = np.exp(log_rho + eta * np.log(scaling))
    rho_new = np.clip(rho_new, rho - move, rho + move)
    rho_new = np.clip(rho_new, rho_min, rho_max)
    return rho_new


class MOC_Optimizer():
    def __init__(
        self,
        cfg: MOC_RAMP_Config,
        tsk: scitopt.mesh.TaskConfig,
    ):
        self.cfg = cfg
        self.tsk = tsk
        if not os.path.exists(self.cfg.dst_path):
            os.makedirs(self.cfg.dst_path)
        # self.tsk.export(self.cfg.dst_path)
        self.cfg.export(self.cfg.dst_path)
        self.tsk.nodes_stats(self.cfg.dst_path)
        
        if os.path.exists(f"{self.cfg.dst_path}/mesh_rho"):
            shutil.rmtree(f"{self.cfg.dst_path}/mesh_rho")
        os.makedirs(f"{self.cfg.dst_path}/mesh_rho")
        if os.path.exists(f"{self.cfg.dst_path}/rho-histo"):
            shutil.rmtree(f"{self.cfg.dst_path}/rho-histo")
        os.makedirs(f"{self.cfg.dst_path}/rho-histo")
        if not os.path.exists(f"{self.cfg.dst_path}/data"):
            os.makedirs(f"{self.cfg.dst_path}/data")

        self.recorder = tools.HistoriesLogger(self.cfg.dst_path)
        self.recorder.add("rho")
        self.recorder.add("rho_diff")
        self.recorder.add("vol_error")
        self.recorder.add("compliance")
        self.recorder.add("dC")
        self.recorder.add("lambda_v")
        self.recorder.add("strain_energy")
        
        self.schedulers = tools.Schedulers(self.cfg.dst_path)
    
    
    def init_schedulers(self):

        cfg = self.cfg
        p_init = cfg.p_init
        vol_frac_init = cfg.vol_frac_init
        move_limit_init = cfg.move_limit_init
        beta_init = cfg.beta_init
        self.schedulers.add(
            "p",
            p_init,
            cfg.p,
            cfg.p_rate,
            cfg.max_iters
        )
        self.schedulers.add(
            "vol_frac",
            vol_frac_init,
            cfg.vol_frac,
            cfg.vol_frac_rate,
            cfg.max_iters
        )
        # print(move_init)
        # print(cfg.move_limit, cfg.move_limit_rate)
        self.schedulers.add(
            "move_limit",
            move_limit_init,
            cfg.move_limit,
            cfg.move_limit_rate,
            cfg.max_iters
        )
        self.schedulers.add(
            "beta",
            beta_init,
            cfg.beta,
            cfg.beta_rate,
            cfg.max_iters
        )
        self.schedulers.export()
    
    def parameterize(self, preprocess=True):
        self.helmholz_solver = filter.HelmholtzFilter.from_defaults(
            self.tsk.mesh, self.cfg.filter_radius, f"{self.cfg.dst_path}/data"
        )
        if preprocess:
            print("preprocessing....")
            # self.helmholz_solver.create_solver()
            self.helmholz_solver.create_LinearOperator()
            print("...end")
        else:
            self.helmholz_solver.create_solver()

    def load_parameters(self):
        self.helmholz_solver = filter.HelmholtzFilter.from_file(
            f"{self.cfg.dst_path}/data"
        )
    
    def optimize(self):
        tsk = self.tsk
        cfg = self.cfg
        
        e_rho = skfem.ElementTetP1()
        basis_rho = skfem.Basis(tsk.mesh, e_rho)
        rho = np.ones(tsk.all_elements.shape)
        iter_begin = 1
        if cfg.restart:
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
            rho[tsk.design_elements] = np.random.uniform(
                0.7, 0.9, size=len(tsk.design_elements)
            )
            # rho[tsk.design_elements] -= np.average(rho[tsk.design_elements])
            # rho[tsk.design_elements] += cfg.vol_frac_init
        print("np.average(rho[tsk.design_elements]):", np.average(rho[tsk.design_elements]))
        self.init_schedulers()
        
        rho_prev = np.zeros_like(rho)
        rho_filtered = np.zeros_like(rho)
        rho_projected = np.zeros_like(rho)
        dH = np.empty_like(rho)
        grad_filtered = np.empty_like(rho)
        dC_drho_projected = np.empty_like(rho)

        dC_drho_sum = np.zeros_like(rho[tsk.design_elements])
        force_list = tsk.force if isinstance(tsk.force, list) else [tsk.force]
        
        mu_p = cfg.mu_p
        mu_d = cfg.mu_d
        mu_i = cfg.mu_i
        vol_error_prev = 0
        vol_integral = 0.0
        vol_error_prev = 0.0
        lambda_v = 0.0
        lambda_v = cfg.lambda_v
        # lambda_decay = cfg.lambda_decay
        for iter in range(iter_begin, cfg.max_iters+iter_begin):
            print(f"iterations: {iter} / {cfg.max_iters}")
            p, vol_frac, beta, move_limit = (
                self.schedulers.values(iter)[k] for k in ['p', 'vol_frac', 'beta', 'move_limit']
            )
            rho_prev[:] = rho[:]
            rho_filtered[:] = self.helmholz_solver.filter(rho)
            rho_filtered[tsk.fixed_elements_in_rho] = 1.0
            
            projection.heaviside_projection_inplace(
                rho_filtered, beta=beta, eta=cfg.beta_eta, out=rho_projected
            )

            dC_drho_sum[:] = 0.0
            strain_energy_sum = 0.0
            for force in force_list:
                compliance, u = solver.compute_compliance_basis_numba(
                    tsk.basis, tsk.free_nodes, tsk.dirichlet_nodes, force,
                    tsk.E0, tsk.Emin, p, tsk.nu0,
                    rho_projected
                )
                strain_energy = composer.compute_strain_energy_numba(
                    u,
                    tsk.basis.element_dofs,
                    tsk.mesh.p,
                    rho_projected,
                    tsk.E0,
                    tsk.Emin,
                    p,
                    tsk.nu0,
                )
                strain_energy_sum += strain_energy
                dC_drho_projected[:] = derivatives.dC_drho_ramp(
                    rho_projected, strain_energy, tsk.E0, tsk.Emin, p
                )

                projection.heaviside_projection_derivative_inplace(
                    rho_filtered, beta=beta, eta=cfg.beta_eta, out=dH
                )
                np.multiply(dC_drho_projected, dH, out=grad_filtered)

                dC_drho_full = self.helmholz_solver.gradient(grad_filtered)
                dC_drho_sum += dC_drho_full[tsk.design_elements]

            dC_drho_sum /= len(force_list)
            strain_energy_sum /= len(force_list)
            
            # rho_e = rho_projected[tsk.design_elements]
            rho_e = rho_filtered[tsk.design_elements]
            vol_error = np.mean(rho_e) - vol_frac
            lambda_v += mu_p * vol_error
            # vol_integral = 0.95 * vol_integral + vol_error
            # d_vol = vol_error - vol_error_prev
            # vol_error_prev = vol_error
            # lambda_v = np.clip(lambda_v, 0.0, 1e4)
            # lambda_v = lambda_decay * lambda_v + mu * vol_error
            # lambda_v += mu_p * vol_error + mu_d * d_vol
            # lambda_v += mu_p * vol_error + mu_i * vol_integral + mu_d * d_vol
            # lambda_v = np.clip(lambda_v, 0.0, 10.0)
            

            dv = np.ones_like(dC_drho_sum) / len(dC_drho_sum)
            # dv = dC / (np.mean(np.abs(dC)) + 1e-8)
            penalty = (lambda_v + mu_p * vol_error) * dv
            # penalty = lambda_v * dv
            # penalty = lambda_v * dC_centered / (np.mean(np.abs(dC_centered)) + 1e-8)

            dC_drho_sum += penalty

            # dC_drho_sum /= np.max(np.abs(dC_drho_sum)) + 1e-8  # optional
            # rho[tsk.design_elements] = oc_update_with_projection(
            #     rho[tsk.design_elements], dC_drho_sum,
            #     move=move_limit, eta=cfg.eta,
            #     rho_min=cfg.rho_min, rho_max=1.0
            # )
            rho[tsk.design_elements] = oc_log_update(
                rho_projected[tsk.design_elements], dC_drho_sum,
                move=move_limit, eta=cfg.eta,
                rho_min=cfg.rho_min, rho_max=1.0
            )
            
            rho[tsk.fixed_elements_in_rho] = 1.0

            # 
            # 
            rho_diff = rho - rho_prev
            # vol_error_prev = vol_error
            # rho_frac = int(len(rho[tsk.design_elements]) * vol_frac)
            
            rho_diff = np.mean(np.abs(rho[tsk.design_elements] - rho_prev[tsk.design_elements]))


            self.recorder.feed_data("rho_diff", rho_diff)
            self.recorder.feed_data("rho", rho_projected)
            self.recorder.feed_data("compliance", compliance)
            self.recorder.feed_data("dC", dC_drho_sum)
            self.recorder.feed_data("lambda_v", lambda_v)
            self.recorder.feed_data("vol_error", vol_error)
            self.recorder.feed_data("strain_energy", strain_energy)
            
            if iter % (cfg.max_iters // self.cfg.record_times) == 0 or iter == 1:
            # if True:
                print(f"Saving at iteration {iter}")
                self.recorder.print()
                # self.recorder_params.print()
                self.recorder.export_progress()
                
                visualization.save_info_on_mesh(
                    tsk,
                    rho_projected, rho_prev,
                    f"{cfg.dst_path}/mesh_rho/info_mesh-{iter}.vtu"
                )
                visualization.export_submesh(
                    tsk, rho_projected, 0.5, f"{cfg.dst_path}/cubic_top.vtk"
                )
                np.savez_compressed(
                    f"{cfg.dst_path}/data/{str(iter).zfill(6)}-rho.npz",
                    rho_design_elements=rho[tsk.design_elements],
                    # compliance=compliance
                )

            # https://qiita.com/fujitagodai4/items/7cad31cc488bbb51f895

        visualization.rho_histo_plot(
            rho_projected[tsk.design_elements],
            f"{self.cfg.dst_path}/rho-histo/last.jpg"
        )

        # threshold = 0.5
        # remove_elements = tsk.design_elements[rho_projected[tsk.design_elements] <= threshold]
        # mask = ~np.isin(tsk.all_elements, remove_elements)
        # kept_elements = tsk.all_elements[mask]
        # Error
        # visualization.export_submesh(tsk, kept_elements, 0.5, f"{self.cfg.dst_path}/cubic_top.vtk")
        # self.export_mesh(rho_projected, "last")
        visualization.export_submesh(
            tsk, rho_projected, 0.5, f"{cfg.dst_path}/cubic_top.vtk"
        )
    
    
if __name__ == '__main__':
    import argparse
    from scitopt.mesh import toy_problem

    parser = argparse.ArgumentParser(
        description=''
    )

    parser.add_argument(
        '--max_iters', '-NI', type=int, default=200, help=''
    )
    parser.add_argument(
        '--filter_radius', '-DR', type=float, default=0.05, help=''
    )
    parser.add_argument(
        '--move_limit_init', '-MLI', type=float, default=0.8, help=''
    )
    parser.add_argument(
        '--move_limit', '-ML', type=float, default=0.2, help=''
    )
    parser.add_argument(
        '--move_limit_rate', '-MLR', type=float, default=5, help=''
    )
    parser.add_argument(
        '--eta', '-ET', type=float, default=1.0, help=''
    )
    parser.add_argument(
        '--record_times', '-RT', type=int, default=20, help=''
    )
    parser.add_argument(
        '--dst_path', '-DP', type=str, default="./result/test0", help=''
    )
    parser.add_argument(
        '--vol_frac_init', '-VI', type=float, default=0.8, help=''
    )
    parser.add_argument(
        '--vol_frac', '-V', type=float, default=0.4, help=''
    )
    parser.add_argument(
        '--vol_frac_rate', '-VFT', type=float, default=20.0, help=''
    )
    parser.add_argument(
        '--p_init', '-PI', type=float, default=1.0, help=''
    )
    parser.add_argument(
        '--p', '-P', type=float, default=3.0, help=''
    )
    parser.add_argument(
        '--p_rate', '-PT', type=float, default=20.0, help=''
    )
    parser.add_argument(
        '--beta_init', '-BI', type=float, default=0.1, help=''
    )
    parser.add_argument(
        '--beta', '-B', type=float, default=5.0, help=''
    )
    parser.add_argument(
        '--beta_rate', '-BR', type=float, default=20.0, help=''
    )
    parser.add_argument(
        '--beta_eta', '-BE', type=float, default=0.5, help=''
    )
    parser.add_argument(
        '--mu_p', '-MUP', type=float, default=100.0, help=''
    )
    parser.add_argument(
        '--mu_d', '-MUD', type=float, default=200.0, help=''
    )
    parser.add_argument(
        '--mu_i', '-MUI', type=float, default=10.0, help=''
    )
    parser.add_argument(
        '--lambda_v', '-LV', type=float, default=1.0, help=''
    )
    parser.add_argument(
        '--restart', '-RS', type=misc.str2bool, default=False, help=''
    )
    parser.add_argument(
        '--restart_from', '-RF', type=int, default=-1, help=''
    )
    parser.add_argument(
        '--task', '-T', type=str, default="toy1", help=''
    )
    args = parser.parse_args()
    

    if args.task == "toy1":
        tsk = toy_problem.toy1()
    elif args.task == "toy1_fine":
        tsk = toy_problem.toy1_fine()
    elif args.task == "toy2":
        tsk = toy_problem.toy2()
    else:
        raise ValueError("task is not indicated")
    
    print("load toy problem")
    
    print("generate OC_RAMP_Config")
    cfg = MOC_RAMP_Config.from_defaults(
        **vars(args)
    )

    print("optimizer")
    optimizer = MOC_Optimizer(cfg, tsk)
    print("parameterize")
    optimizer.parameterize(preprocess=True)
    # optimizer.parameterize(preprocess=False)
    # optimizer.load_parameters()
    print("optimize")
    optimizer.optimize()
    # optimizer.optimize_org()