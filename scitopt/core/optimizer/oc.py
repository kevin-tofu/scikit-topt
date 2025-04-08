import os
from typing import Literal
import inspect
import math
import shutil
import json
from dataclasses import dataclass, asdict
import numpy as np
import scipy
import scipy.sparse.linalg as spla
from numba import njit
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
class OC_Config():
    dst_path: str = "./result"
    interpolation: Literal["SIMP", "RAMP"] = "SIMP"
    record_times: int=20
    max_iters: int=200
    p_init: float = 1.0
    p: float = 5.0
    p_rate: float = 20.0
    vol_frac_init: float = 0.8
    vol_frac: float = 0.4  # the maximum valume ratio
    vol_frac_rate: float = 20.0
    beta_init: float = 1.0
    beta: float = 3
    beta_rate: float = 12.
    beta_eta: float = 0.50
    filter_radius: float = 0.40
    eta: float = 0.5
    rho_min: float = 0.05
    rho_max: float = 1.0
    move_limit_init: float = 0.20
    move_limit: float = 0.15
    move_limit_rate: float = 5.0
    bisec_lambda_lower: float=1e-5
    bisec_lambda_upper: float=500
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



def bisection_with_projection(
    dC, rho_e, rho_min, rho_max, move_limit,
    eta, eps, vol_frac,
    beta, beta_eta,
    scaling_rate, rho_candidate, tmp_lower, tmp_upper,
    max_iter=100, tolerance=1e-4,
    l1 = 1.0,
    l2 = 1000.0
):
    # for _ in range(100):
    # while abs(l2 - l1) <= tolerance * (l1 + l2) / 2.0:
    # while abs(l2 - l1) > tolerance * (l1 + l2) / 2.0:
    while abs(l2 - l1) > tolerance:
        lmid = 0.5 * (l1 + l2)
        # print("lmid:", lmid)
        # if abs(l2 - l1) <= tolerance * (l1 + l2) / 2.0:
        #     break
        
        # 0 < lmid 
        np.negative(dC, out=scaling_rate)
        scaling_rate /= (lmid + eps)
        sign = np.sign(scaling_rate)
        np.abs(scaling_rate, out=scaling_rate)
        np.power(scaling_rate, eta, out=scaling_rate)
        scaling_rate *= sign

        # Clip
        np.clip(scaling_rate, 0.8, 1.2, out=scaling_rate)
        # np.clip(scaling_rate, 0.5, 1.5, out=scaling_rate)
        # np.clip(scaling_rate, 0.1, 3.0, out=scaling_rate)
        # np.clip(scaling_rate, 0.1, 5.0, out=scaling_rate)
        
        np.multiply(rho_e, scaling_rate, out=rho_candidate)
        np.maximum(rho_e - move_limit, rho_min, out=tmp_lower)
        np.minimum(rho_e + move_limit, rho_max, out=tmp_upper)
        np.clip(rho_candidate, tmp_lower, tmp_upper, out=rho_candidate)

        projection.heaviside_projection_inplace(
            rho_candidate, beta=beta, eta=beta_eta, out=rho_candidate
        )
        vol_error = np.mean(rho_candidate) - vol_frac
        if abs(vol_error) < 1e-6:
            break
        if vol_error > 0:
            l1 = lmid
        else:
            l2 = lmid
        
        # if abs(l2 - l1) < 1e-6:
        #     l1 *= 0.9
        #     l2 *= 1.1
            
    return rho_candidate, lmid, vol_error




class OC_Optimizer():
    def __init__(
        self,
        cfg: OC_Config,
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
        self.recorder.add("dC_drho_dirichlet")
        self.recorder.add("lambda_v", ylog=True)
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

        cfg = self.cfg
        tsk = self.tsk
        
        @njit
        def compute_safe_dC(dC):
            mean_val = np.mean(dC)
            dC -= mean_val
            norm = np.percentile(np.abs(dC), 95) + 1e-8
            dC /= norm
            
        # @njit
        # def compute_safe_dC(dC, min_threshold=0.05):
        #     n = dC.size
        #     mean_val = np.sum(dC) / n
        #     for i in range(n):
        #         dC[i] -= mean_val
        #     max_abs = 0.0
        #     for i in range(n):
        #         abs_val = abs(dC[i])
        #         if abs_val > max_abs:
        #             max_abs = abs_val
        #     norm = max_abs + 1e-8
        #     for i in range(n):
        #         dC[i] /= norm
        #         if abs(dC[i]) < min_threshold:
        #             dC[i] = min_threshold if dC[i] >= 0 else -min_threshold

        #     return dC



        # @njit
        # def compute_safe_dC(dC):
        #     norm = np.percentile(np.abs(dC), 95) + 1e-8
        #     if norm > 0:
        #         dC[:] /= norm

        # @njit
        # def compute_safe_dC(dC):
        #     norm = np.max(np.abs(dC)) + 1e-8
        #     dC[:] /= norm
        #     dC[:] = np.maximum(dC, 1e-4)
        

        rho = np.ones_like(tsk.all_elements)
        rho = rho * cfg.vol_frac if cfg.vol_frac_rate < 0 else rho * cfg.vol_frac_init
        # rho = np.ones(tsk.all_elements.shape) * 0.5
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
            # rho[tsk.design_elements] = np.minimum(
            #     np.random.uniform(
            #         cfg.vol_frac_init - 0.2,
            #         cfg.vol_frac_init + 0.2,
            #         size=len(tsk.design_elements)
            #     ),
            #     1.0
            # )
            pass
        rho[tsk.dirichlet_force_elements] = 1.0
        print("np.average(rho[tsk.design_elements]):", np.average(rho[tsk.design_elements]))
        
        self.init_schedulers()
        eta = cfg.eta
        rho_min = cfg.rho_min
        rho_max = 1.0
        tolerance = 1e-6
        eps = 1e-6

        rho_prev = np.zeros_like(rho)

        rho_filtered = np.zeros_like(rho)
        rho_projected = np.zeros_like(rho)
        dH = np.empty_like(rho)
        grad_filtered = np.empty_like(rho)
        dC_drho_projected = np.empty_like(rho)

        # dC_drho_ave = np.zeros_like(rho)
        dC_drho_full = np.zeros_like(rho)
        dC_drho_dirichlet = np.zeros_like(rho[tsk.dirichlet_elements])
        dC_drho_ave = np.zeros_like(rho[tsk.design_elements])
        scaling_rate = np.empty_like(rho[tsk.design_elements])
        rho_candidate = np.empty_like(rho[tsk.design_elements])
        tmp_lower = np.empty_like(rho[tsk.design_elements])
        tmp_upper = np.empty_like(rho[tsk.design_elements])
        force_list = tsk.force if isinstance(tsk.force, list) else [tsk.force]
        rho_min_boundary = 0.2
        
        if cfg.interpolation == "SIMP":
        # if False:
            density_interpolation = composer.simp_interpolation_numba
            dC_drho_func = derivatives.dC_drho_simp
        elif cfg.interpolation == "RAMP":
            density_interpolation = composer.ramp_interpolation_numba
            dC_drho_func = derivatives.dC_drho_ramp
        else:
            raise ValueError("should be SIMP or RAMP")


        for iter_local, iter in enumerate(range(iter_begin, cfg.max_iters + iter_begin)):
            print(f"iterations: {iter} / {cfg.max_iters}")
            p, vol_frac, beta, move_limit = (
                self.schedulers.values(iter)[k] for k in ['p', 'vol_frac', 'beta', 'move_limit']
            )
            print(f"p {p:.4f}, vol_frac {vol_frac:.4f}, beta {beta:.4f}, move_limit {move_limit:.4f}")

            rho_prev[:] = rho[:]
            rho_filtered[:] = self.helmholz_solver.filter(rho)
            rho_filtered[tsk.force_elements] = 1.0
            # rho_filtered[tsk.dirichlet_force_elements] = 1.0
            # rho_filtered[tsk.dirichlet_force_elements] = 1.0
            # if iter_local < 120:
            #     rho_filtered[tsk.dirichlet_force_elements] = 1.0
            # else:
            #     rho_filtered[tsk.fixed_elements_in_rho] = 1.0

            projection.heaviside_projection_inplace(
                rho_filtered, beta=beta, eta=cfg.beta_eta, out=rho_projected
            )
            # if True:
            if False:
                rho_projected[tsk.dirichlet_elements] = np.maximum(
                    rho_projected[tsk.dirichlet_elements], rho_min_boundary
                )

            dC_drho_ave[:] = 0.0
            dC_drho_dirichlet[:] = 0.0
            strain_energy_sum = 0.0
            compliance_avg = 0.0
            # dC_drho_dirichlet_scaling = True if iter_local < 50 else False
            for force in force_list:
                compliance, u = solver.compute_compliance_basis_numba(
                    tsk.basis, tsk.free_nodes, tsk.dirichlet_nodes, force,
                    tsk.E0, tsk.Emin, p, tsk.nu0,
                    rho_projected,
                    # rho_filtered,
                    density_interpolation
                )
                compliance_avg += compliance
                strain_energy = composer.compute_strain_energy_numba(
                    u,
                    # tsk.basis.element_dofs[:, tsk.design_elements],
                    tsk.basis.element_dofs,
                    tsk.mesh.p,
                    rho_projected,
                    # rho_filtered,
                    tsk.E0,
                    tsk.Emin,
                    p,
                    tsk.nu0,
                )
                strain_energy_sum += strain_energy
                dC_drho_projected[:] = dC_drho_func(
                    # rho_filtered,
                    rho_projected,
                    strain_energy, tsk.E0, tsk.Emin, p
                )
                projection.heaviside_projection_derivative_inplace(
                    rho_filtered,
                    beta=beta, eta=cfg.beta_eta, out=dH
                )
                np.multiply(dC_drho_projected, dH, out=grad_filtered)
                dC_drho_full[:] = self.helmholz_solver.gradient(grad_filtered)
                # if dC_drho_dirichlet_scaling:
                #     dC_drho_full[tsk.dirichlet_elements] *= 10.0
                dC_drho_ave += dC_drho_full[tsk.design_elements]
                # dC_drho_ave += self.helmholz_solver.gradient(grad_filtered)
                dC_drho_dirichlet += dC_drho_full[tsk.dirichlet_elements]

            dC_drho_ave /= len(force_list)
            dC_drho_dirichlet /= len(force_list)
            strain_energy_sum /= len(force_list)
            compliance_avg /= len(force_list)

            compute_safe_dC(dC_drho_ave)
            # dC_drho_ave[:] = self.helmholz_solver.filter(dC_drho_ave)

            rho_e = rho_projected[tsk.design_elements]
            # bisection_with_projection
            # bisection_with_projection_bi
            rho_candidate, lmid, vol_error = bisection_with_projection(
                dC_drho_ave,
                # dC_drho_ave[tsk.design_elements],
                rho_e, cfg.rho_min, cfg.rho_max, move_limit,
                cfg.eta, eps, vol_frac,
                beta, cfg.beta_eta,
                scaling_rate, rho_candidate, tmp_lower, tmp_upper,
                max_iter=100, tolerance=1e-4,
                l1 = cfg.bisec_lambda_lower,
                l2 = cfg.bisec_lambda_upper
            )
            print(
                f"λ: {lmid:.4e}, vol_error: {vol_error:.4f}, mean(rho): {np.mean(rho_candidate):.4f}"
            )
            rho[tsk.design_elements] = rho_candidate
            # if iter_local < 120:
            #     rho[tsk.dirichlet_force_elements] = 1.0
            # else:
            #     rho[tsk.force_elements] = 1.0
            # rho[tsk.force_elements] = 1.0
            rho_diff = np.mean(np.abs(rho[tsk.design_elements] - rho_prev[tsk.design_elements]))


            self.recorder.feed_data("rho", rho_projected[tsk.design_elements])
            self.recorder.feed_data("dC_drho_dirichlet", dC_drho_dirichlet)
            self.recorder.feed_data("scaling_rate", scaling_rate)
            self.recorder.feed_data("compliance", compliance_avg)
            self.recorder.feed_data("dC", dC_drho_ave)
            self.recorder.feed_data("lambda_v", lmid)
            self.recorder.feed_data("vol_error", vol_error)
            self.recorder.feed_data("strain_energy", strain_energy)
            
            
            if iter % (cfg.max_iters // self.cfg.record_times) == 0 or iter == 1:
            # if True:
                print(f"Saving at iteration {iter}")
                # self.recorder.print()
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
                    rho_projected_design_elements=rho_projected[tsk.design_elements],
                )

            # https://qiita.com/fujitagodai4/items/7cad31cc488bbb51f895

        visualization.rho_histo_plot(
            rho_projected[tsk.design_elements],
            f"{self.cfg.dst_path}/rho-histo/last.jpg"
        )
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
        '--interpolation', '-I', type=str, default="SIMP", help=''
    )
    parser.add_argument(
        '--max_iters', '-NI', type=int, default=200, help=''
    )
    parser.add_argument(
        '--filter_radius', '-DR', type=float, default=0.8, help=''
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
        '--bisec_lambda_lower', '-BSL', type=float, default=-20.0, help=''
    )
    parser.add_argument(
        '--bisec_lambda_upper', '-BSH', type=float, default=20.0, help=''
    )
    parser.add_argument(
        '--restart', '-RS', type=misc.str2bool, default=False, help=''
    )
    parser.add_argument(
        '--restart_from', '-RF', type=int, default=-1, help=''
    )
    parser.add_argument(
        '--rho_min', '-RM', type=float, default=0.01, help=''
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
        tsk = toy_problem.toy_msh(args.task)
    
    print("load toy problem")
    
    print("generate OC_Config")
    cfg = OC_Config.from_defaults(
        **vars(args)
    )
    
    print("optimizer")
    optimizer = OC_Optimizer(cfg, tsk)
    print("parameterize")
    optimizer.parameterize(preprocess=True)
    # optimizer.parameterize(preprocess=False)
    # optimizer.load_parameters()
    print("optimize")
    optimizer.optimize()
    # optimizer.optimize_org()
