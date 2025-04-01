import os
import inspect
import math
import shutil
import json
from dataclasses import dataclass, asdict
import numpy as np
import scipy
import scipy.sparse.linalg as spla
import skfem
import meshio
from scitopt import mesh
from scitopt import tools
from scitopt.core import derivatives, projection
from scitopt.core import visualization
from scitopt.fea import solver
from scitopt import filter
from scitopt.fea import composer


@dataclass
class OC_RAMP_Config():
    dst_path: str = "./result"
    record_times: int=20
    max_iters: int=200
    p: float = 3
    p_rate: float = 20.0
    vol_frac: float = 0.4  # the maximum valume ratio
    vol_frac_rate: float = 20.0
    beta: float = 8
    beta_rate: float = 20.
    beta_eta: float = 0.3
    filter_radius: float = 0.05
    eta: float = 0.3
    rho_min: float = 1e-3
    rho_max: float = 1.0
    move_limit: float = 0.2
    move_limit_rate: float = 20.0
    

    @classmethod
    def from_defaults(cls, **args):
        sig = inspect.signature(cls)
        valid_keys = sig.parameters.keys()
        filtered_args = {k: v for k, v in args.items() if k in valid_keys}
        return cls(**filtered_args)


    def export(self, path: str):
        with open(f"{path}/cfg.json", "w") as f:
            json.dump(asdict(self), f, indent=2)


class OC_Optimizer():
    def __init__(
        self,
        cfg: OC_RAMP_Config,
        prb: mesh.TaskConfig,
    ):
        self.prb = prb
        self.cfg = cfg
        if not os.path.exists(self.cfg.dst_path):
            os.makedirs(self.cfg.dst_path)
        # self.prb.export(self.cfg.dst_path)
        self.cfg.export(self.cfg.dst_path)
        self.prb.nodes_stats(self.cfg.dst_path)
        
        if os.path.exists(f"{self.cfg.dst_path}/mesh_rho"):
            shutil.rmtree(f"{self.cfg.dst_path}/mesh_rho")
        os.makedirs(f"{self.cfg.dst_path}/mesh_rho")
        if os.path.exists(f"{self.cfg.dst_path}/rho-histo"):
            shutil.rmtree(f"{self.cfg.dst_path}/rho-histo")
        os.makedirs(f"{self.cfg.dst_path}/rho-histo")
        if not os.path.exists(f"{self.cfg.dst_path}/matrices"):
            os.makedirs(f"{self.cfg.dst_path}/matrices")

        self.recorder = tools.HistoriesLogger(self.cfg.dst_path)
        self.recorder.add("rho")
        self.recorder.add("rho_diff")
        self.recorder.add("lambda_v")
        self.recorder.add("vol_error")
        self.recorder.add("compliance")
        self.recorder.add("dC")
        self.recorder.add("scaling_rate")
        self.recorder.add("strain_energy")
        # self.recorder_params = self.history.HistoriesLogger(self.cfg.dst_path)
        # self.recorder_params.add("p")
        # self.recorder_params.add("vol_frac")
        # self.recorder_params.add("beta")
        # self.recorder_params.add("move_limit")
        
        self.schedulers = tools.Schedulers(self.cfg.dst_path)
    
    
    def init_schedulers(
        self,
        p_init, vol_frac_init, move_init, beta_init
    ):
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
            move_init,
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
            self.prb.mesh, self.cfg.filter_radius, f"{self.cfg.dst_path}/matrices"
        )
        if preprocess:
            print("preprocessing....")
            # self.helmholz_solver.create_solver()
            self.helmholz_solver.create_LinearOperator()
            print("...end")


    def load_parameters(self):
        self.helmholz_solver = filter.HelmholtzFilter.from_file(
            f"{self.cfg.dst_path}/matrices"
        )
    
    
    def optimize(self):
        
        self.init_schedulers(1.0, 0.8, 0.8, cfg.beta / 10.0)
        e_rho = skfem.ElementTetP1()
        # basis_rho = skfem.Basis(prb.mesh, e_rho)
        rho = np.ones(prb.all_elements.shape)
        # rho[prb.design_elements] = 0.95
        # rho[prb.design_elements] = cfg.vol_frac
        rho[prb.design_elements] = np.random.uniform(
            0.5, 0.8, size=len(prb.design_elements)
        )
        rho_projected = projection.heaviside_projection(
            rho, beta=cfg.beta / 10.0, eta=cfg.beta_eta
        )
        p, vol_frac, beta, move_limit = (
            self.schedulers.values(0)[k] for k in ['p', 'vol_frac', 'beta', 'move_limit']
        )
        eta = cfg.eta
        rho_min = cfg.rho_min
        rho_max = 1.0
        tolerance = 1e-4
        eps = 1e-6
        rho_prev = np.zeros_like(rho)
        force_list = prb.force if isinstance(prb.force, list) else [prb.force]
        
        for iter in range(1, cfg.max_iters+1):
            print(f"iterations: {iter} / {cfg.max_iters}")
            p, vol_frac, beta, move_limit = (
                self.schedulers.values(iter)[k] for k in ['p', 'vol_frac', 'beta', 'move_limit']
            )
            print(
                f"p {p:0.4f}, vol_frac {vol_frac:0.4f}, beta {beta:0.4f}, move_limit {move_limit:0.4f}"
            )
            
            rho_prev[:] = rho[:]
            rho_filtered = self.helmholz_solver.filter(rho)
            rho_filtered[prb.fixed_elements_in_rho] = 1.0
            rho_projected = projection.heaviside_projection(
                rho_filtered, beta=beta, eta=cfg.beta_eta
            )
            dC_drho_sum = 0
            strain_energy_sum = 0
            for force in force_list:
                compliance, u = solver.computer_compliance_simp_basis(
                    prb.basis, prb.free_nodes, prb.dirichlet_nodes, force,
                    prb.E0, prb.Emin, p, prb.nu0, rho
                )
            
                # Compute strain energy and obtain derivatives
                strain_energy = composer.compute_strain_energy(
                    u,
                    prb.basis.element_dofs,
                    prb.basis, rho_projected,
                    prb.E0, prb.Emin, p, prb.nu0
                )
                strain_energy_sum += strain_energy
                dC_drho_projected = derivatives.dC_drho_ramp(
                    rho_projected, strain_energy, prb.E0, prb.Emin, p
                )
                dH = projection.heaviside_projection_derivative(
                    rho_filtered, beta=beta, eta=cfg.beta_eta
                )
                grad_filtered = dC_drho_projected * dH
                dC_drho = self.helmholz_solver.gradient(grad_filtered)
                dC_drho = dC_drho[prb.design_elements]
                dC_drho_sum += dC_drho
            
            dC_drho_sum /= len(force_list)
            strain_energy_sum /= len(force_list)

            # 
            # Correction with Lagrange multipliers Bisection Method
            # 
            safe_dC = dC_drho_sum - np.mean(dC_drho_sum)
            norm = np.percentile(np.abs(safe_dC), 95) + 1e-8
            safe_dC = safe_dC / norm
            # safe_dC = safe_dC / (np.max(np.abs(safe_dC)) + 1e-8)
            # safe_dC = np.clip(safe_dC, -1.0, 1.0)
            # safe_dC = np.clip(safe_dC, -5, 5)

            
            rho_e = rho_projected[prb.design_elements].copy()
            # l1, l2 = 1e-9, 1e4
            l1, l2 = 1e-9, 500
            lmid = 0.5 * (l1 + l2)
            while abs(l2 - l1) > tolerance * (l1 + l2) / 2.0:
            # while (l2 - l1) / (0.5 * (l1 + l2) + eps) > tolerance:
            # while abs(vol_error) > 1e-2:
                lmid = 0.5 * (l1 + l2)
                scaling_rate = (- safe_dC / (lmid + eps)) ** eta
                scaling_rate = np.clip(scaling_rate, 0.5, 1.5)

                rho_candidate = np.clip(
                    rho_e * scaling_rate,
                    np.maximum(rho_e - move_limit, rho_min),
                    np.minimum(rho_e + move_limit, rho_max)
                )
                rho_candidate_projected = projection.heaviside_projection(
                    rho_candidate, beta=beta, eta=cfg.beta_eta
                )
                vol_error = np.mean(rho_candidate_projected) - vol_frac
                if vol_error > 0:
                    l1 = lmid
                else:
                    l2 = lmid

            rho[prb.design_elements] = rho_candidate
            rho_diff = rho - rho_prev

            self.recorder.feed_data("rho_diff", rho_diff[prb.design_elements])
            self.recorder.feed_data("scaling_rate", scaling_rate)
            self.recorder.feed_data("rho", rho_projected[prb.design_elements])
            self.recorder.feed_data("compliance", compliance)
            self.recorder.feed_data("dC", dC_drho)
            self.recorder.feed_data("lambda_v", lmid)
            self.recorder.feed_data("vol_error", vol_error)
            self.recorder.feed_data("strain_energy", strain_energy)
            
            # if np.sum(np.abs(rho_diff)) < 1e-3:
            #     noise_strength = 0.03
            #     # rho[prb.design_elements] += np.random.uniform(
            #     #     -noise_strength, noise_strength, size=prb.design_elements.shape
            #     # )
            #     rho[prb.design_elements] += -safe_dC / (np.abs(safe_dC).max() + 1e-8) * 0.05 \
            #         + np.random.normal(0, noise_strength, size=prb.design_elements.shape)
            #     rho[prb.design_elements] = np.clip(rho[prb.design_elements], cfg.rho_min, cfg.rho_max)

            if iter % (cfg.max_iters // self.cfg.record_times) == 0 or iter == 1:
            # if True:
                print(f"Saving at iteration {iter}")
                self.recorder.print()
                # self.recorder_params.print()
                self.recorder.export_progress()
                visualization.save_info_on_mesh(
                    prb,
                    rho_projected, rho_prev,
                    f"{cfg.dst_path}/mesh_rho/info_mesh-{iter}.vtu"
                )
                visualization.export_submesh(
                    prb, rho_projected, 0.5, f"{cfg.dst_path}/cubic_top.vtk"
                )

            # https://qiita.com/fujitagodai4/items/7cad31cc488bbb51f895

        visualization.rho_histo_plot(
            rho_projected[prb.design_elements],
            f"{self.cfg.dst_path}/rho-histo/last.jpg"
        )

        threshold = 0.5
        remove_elements = prb.design_elements[rho_projected[prb.design_elements] <= threshold]
        mask = ~np.isin(prb.all_elements, remove_elements)
        kept_elements = prb.all_elements[mask]
        visualization.export_submesh(prb, kept_elements, 0.5, f"{self.cfg.dst_path}/cubic_top.vtk")
        # self.export_mesh(rho_projected, "last")


if __name__ == '__main__':

    import argparse
    from scitopt.mesh import toy_problem
    
    parser = argparse.ArgumentParser(
        description=''
    )
    parser.add_argument(
        '--p', '-P', type=float, default=3.0, help=''
    )
    parser.add_argument(
        '--vol_frac', '-V', type=float, default=0.4, help=''
    )
    parser.add_argument(
        '--learning_rate', '-LR', type=float, default=0.1, help=''
    )
    parser.add_argument(
        '--max_iters', '-NI', type=int, default=200, help=''
    )
    parser.add_argument(
        '--filter_radius', '-DR', type=float, default=0.05, help=''
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
        '--problem', '-PM', type=str, default="toy2", help=''
    )
    parser.add_argument(
        '--vol_frac_rate', '-VFT', type=float, default=20.0, help=''
    )
    parser.add_argument(
        '--p_rate', '-PT', type=float, default=20.0, help=''
    )
    parser.add_argument(
        '--beta', '-B', type=float, default=100.0, help=''
    )
    parser.add_argument(
        '--beta_rate', '-BR', type=float, default=20.0, help=''
    )
    args = parser.parse_args()
    

    if args.problem == "toy":
        prb = toy_problem.toy()
    elif args.problem == "toy2":
        prb = toy_problem.toy2()
    
    print("load toy problem")
    
    print("generate OC_RAMP_Config")
    cfg = OC_RAMP_Config.from_defaults(
        **vars(args)
    )
    
    print("optimizer")
    optimizer = OC_Optimizer(cfg, prb)
    print("parameterize")
    optimizer.parameterize(preprocess=True)
    # optimizer.load_parameters()
    print("optimize")
    optimizer.optimize()