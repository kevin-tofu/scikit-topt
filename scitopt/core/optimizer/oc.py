from typing import Literal, Callable
from dataclasses import dataclass
import numpy as np
import scitopt
from scitopt.core import projection
from scitopt import filter
from scitopt.core import misc
from scitopt.core.optimizer import common


@dataclass
class OC_Config(common.Sensitivity_Config):
    interpolation: Literal["SIMP"] = "SIMP"
    eta_init: float = 0.1
    eta: float = 0.5
    eta_step: int = 3
    

def bisection_with_projection(
    dC, rho_e, rho_min, rho_max, move_limit,
    eta, eps, vol_frac,
    beta, beta_eta,
    scaling_rate, rho_candidate,
    tmp_lower, tmp_upper,
    elements_volume, elements_volume_sum,
    max_iter=100, tolerance=1e-4,
    l1 = 1e-3,
    l2 = 1e+3
):
    # for _ in range(100):
    # while abs(l2 - l1) <= tolerance * (l1 + l2) / 2.0:
    # while abs(l2 - l1) > tolerance * (l1 + l2) / 2.0:
    while abs(l2 - l1) > tolerance:
        lmid = 0.5 * (l1 + l2)
        np.negative(dC, out=scaling_rate)
        scaling_rate /= (lmid + eps)
        np.power(scaling_rate, eta, out=scaling_rate)

        # Clip
        np.clip(scaling_rate, 0.8, 1.2, out=scaling_rate)
        
        np.multiply(rho_e, scaling_rate, out=rho_candidate)
        np.maximum(rho_e - move_limit, rho_min, out=tmp_lower)
        np.minimum(rho_e + move_limit, rho_max, out=tmp_upper)
        np.clip(rho_candidate, tmp_lower, tmp_upper, out=rho_candidate)

        # 
        # filter might be needed here
        # 
        projection.heaviside_projection_inplace(
            rho_candidate, beta=beta, eta=beta_eta, out=rho_candidate
        )
        
        # vol_error = np.mean(rho_candidate) - vol_frac
        vol_error = np.sum(
            rho_candidate * elements_volume
        ) / elements_volume_sum - vol_frac
        
        if abs(vol_error) < 1e-6:
            break
        if vol_error > 0:
            l1 = lmid
        else:
            l2 = lmid
            
    return lmid, vol_error


class OC_Optimizer(common.Sensitivity_Analysis):
    def __init__(
        self,
        cfg: OC_Config,
        tsk: scitopt.mesh.TaskConfig,
    ):
        super().__init__(cfg, tsk)
        self.recorder.add("-dC", plot_type="min-max-mean-std", ylog=True)
        self.recorder.add("lmid", ylog=False) # True
            
    
    def init_schedulers(self, export: bool=True):
        super().init_schedulers(False)
        self.schedulers.add(
            "eta",
            self.cfg.eta_init,
            self.cfg.eta,
            self.cfg.eta_step,
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
        beta: float,
        tmp_lower: np.ndarray,
        tmp_upper: np.ndarray,
        percentile: float,
        interploation: Callable,
        elements_volume_design: np.ndarray,
        elements_volume_design_sum: float,
        vol_frac: float
    ):
        cfg = self.cfg
        tsk = self.tsk
        eps = 1e-6
        
        scale = np.percentile(np.abs(dC_drho_ave), percentile)
        # scale = max(scale, np.mean(np.abs(dC_drho_ave)), 1e-4)
        # scale = np.median(np.abs(dC_drho_full[tsk.design_elements]))
        self.running_scale = 0.6 * self.running_scale + (1 - 0.6) * scale if iter_loop > 1 else scale
        dC_drho_ave /= (self.running_scale + eps)
        print(f"dC_drho_ave-scaled min:{dC_drho_ave.min()} max:{dC_drho_ave.max()}")
        print(f"dC_drho_ave-scaled ave:{np.mean(dC_drho_ave)} sdv:{np.std(dC_drho_ave)}")
        rho_e = rho_projected[tsk.design_elements]

        lmid, vol_error = bisection_with_projection(
            dC_drho_ave,
            rho_e, cfg.rho_min, cfg.rho_max, move_limit,
            eta, eps, vol_frac,
            beta, cfg.beta_eta,
            scaling_rate, rho_candidate,
            tmp_lower, tmp_upper,
            elements_volume_design, elements_volume_design_sum,
            max_iter=1000, tolerance=1e-5,
            l1 = cfg.lambda_lower,
            l2 = cfg.lambda_upper
        )
        print(
            f"λ: {lmid:.4e}, vol_error: {vol_error:.4f}, mean(rho): {np.mean(rho_candidate):.4f}"
        )
        self.recorder.feed_data("lmid", lmid)
        self.recorder.feed_data("vol_error", vol_error)
        self.recorder.feed_data("-dC", -dC_drho_ave)


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
        '--filter_radius_init', '-FRI', type=float, default=0.2, help=''
    )
    parser.add_argument(
        '--filter_radius', '-FR', type=float, default=0.05, help=''
    )
    parser.add_argument(
        '--filter_radius_step', '-FRS', type=int, default=3, help=''
    )
    parser.add_argument(
        '--move_limit_init', '-MLI', type=float, default=0.8, help=''
    )
    parser.add_argument(
        '--move_limit', '-ML', type=float, default=0.2, help=''
    )
    parser.add_argument(
        '--move_limit_step', '-MLR', type=int, default=5, help=''
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
        '--vol_frac_step', '-VFT', type=int, default=2, help=''
    )
    parser.add_argument(
        '--p_init', '-PI', type=float, default=1.0, help=''
    )
    parser.add_argument(
        '--p', '-P', type=float, default=3.0, help=''
    )
    parser.add_argument(
        '--p_step', '-PRT', type=int, default=2, help=''
    )
    parser.add_argument(
        '--beta_init', '-BI', type=float, default=0.1, help=''
    )
    parser.add_argument(
        '--beta', '-B', type=float, default=5.0, help=''
    )
    parser.add_argument(
        '--beta_step', '-BR', type=float, default=20.0, help=''
    )
    parser.add_argument(
        '--percentile_init', '-PTI', type=float, default=60, help=''
    )
    parser.add_argument(
        '--percentile_step', '-PTR', type=int, default=2, help=''
    )
    parser.add_argument(
        '--percentile', '-PT', type=float, default=90, help=''
    )
    parser.add_argument(
        '--eta_init', '-ETI', type=float, default=0.01, help=''
    )
    parser.add_argument(
        '--eta_step', '-ETR', type=float, default=-1.0, help=''
    )
    parser.add_argument(
        '--eta', '-ET', type=float, default=0.3, help=''
    )
    parser.add_argument(
        '--beta_eta', '-BE', type=float, default=0.5, help=''
    )
    parser.add_argument(
        '--lambda_lower', '-BSL', type=float, default=1e-4, help=''
    )
    parser.add_argument(
        '--lambda_upper', '-BSH', type=float, default=1e+2, help=''
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
    parser.add_argument(
        '--export_img', '-EI', type=misc.str2bool, default=False, help=''
    )
    parser.add_argument(
        '--design_dirichlet', '-DD', type=misc.str2bool, default=True, help=''
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
    optimizer.parameterize()
    # optimizer.parameterize(preprocess=False)
    # optimizer.load_parameters()
    print("optimize")
    optimizer.optimize()
    # optimizer.optimize_org()
