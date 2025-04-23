from dataclasses import dataclass
import numpy as np
import scitopt
from scitopt.core import misc
from scitopt.core.optimizer import common


@dataclass
class KKT_Config(common.Sensitivity_Config):
    mu_p: float = 2.0
    lambda_v: float = 0.1
    lambda_decay: float = 0.95


# log(x) = -0.4   →   x ≈ 0.670
# log(x) = -0.3   →   x ≈ 0.741
# log(x) = -0.2   →   x ≈ 0.819
# log(x) = -0.1   →   x ≈ 0.905
# log(x) =  0.0   →   x =  1.000
# log(x) = +0.1   →   x ≈ 1.105
# log(x) = +0.2   →   x ≈ 1.221
# log(x) = +0.3   →   x ≈ 1.350
# log(x) = +0.4   →   x ≈ 1.492


def kkt_moc_log_update(
    rho,
    dC, lambda_v, scaling_rate,
    eta, move_limit,
    tmp_lower, tmp_upper,
    rho_min: float, rho_max: float,
    percentile: float,
    interpolation: str
):
    """
    In-place version of the modified OC update (log-space),
    computing dL = dC + lambda_v inside the function.

    Parameters:
    - rho: np.ndarray, design variables (will be updated in-place)
    - dC: np.ndarray, compliance sensitivity (usually negative)
    - lambda_v: float, Lagrange multiplier for volume constraint
    - move: float, maximum allowed change per iteration
    - eta: float, learning rate
    - rho_min: float, minimum density
    - rho_max: float, maximum density
    - tmp_lower, tmp_upper, scaling_rate: work arrays (same shape as rho)
    """

    eps = 1e-8

    # Compute dL = dC + lambda_v
    # np.copyto(scaling_rate, dC)
    # scaling_rate += lambda_v
    # norm = np.percentile(np.abs(scaling_rate), percentile) + 1e-8
    # scaling_rate /= norm

    # Normalize: subtract mean
    # print(f"interpolation: {interpolation}")
    np.copyto(scaling_rate, dC)
    if interpolation == "SIMP":
        norm = np.percentile(np.abs(dC), percentile) + 1e-8
    elif interpolation == "RAMP":
        scaling_rate -= np.mean(dC)
        percentile_value = np.percentile(np.abs(scaling_rate), percentile)
        # norm = max(percentile_value, 1e-4)
        norm = percentile_value
        # norm = max(np.abs(scaling_rate), 1e-4)
        # print(f"percentile_value: {percentile_value}, norm: {norm}")
    else:
        raise ValueError("should be SIMP/RAMP")
    np.divide(scaling_rate, norm, out=scaling_rate)
    # scaling_rate /= norm
    scaling_rate += lambda_v
    # Optional clipping of scaled gradient
    # np.clip(scaling_rate, -0.3, 0.3, out=scaling_rate)
    # np.clip(scaling_rate, -0.5, 0.5, out=scaling_rate)
    np.clip(scaling_rate, -1.0, 1.0, out=scaling_rate)

    # Ensure rho is in [rho_min, 1.0] before log
    np.clip(rho, rho_min, 1.0, out=rho)

    # tmp_lower = log(rho)
    np.log(rho, out=tmp_lower)

    # tmp_upper = exp(log(rho)) = rho
    np.exp(tmp_lower, out=tmp_upper)

    # tmp_upper = log(1 + move / rho)
    np.divide(move_limit, tmp_upper, out=tmp_upper)
    np.add(tmp_upper, 1.0, out=tmp_upper)
    np.log(tmp_upper, out=tmp_upper)

    # tmp_lower = lower bound in log-space
    np.subtract(tmp_lower, tmp_upper, out=tmp_lower)

    # tmp_upper = upper bound in log-space
    np.add(tmp_lower, 2.0 * tmp_upper, out=tmp_upper)

    # rho = log(rho)
    np.log(rho, out=rho)

    # Update in log-space
    rho -= eta * scaling_rate

    # Apply move limits
    np.clip(rho, tmp_lower, tmp_upper, out=rho)

    # Convert back to real space
    np.exp(rho, out=rho)

    # Final clipping
    np.clip(rho, rho_min, rho_max, out=rho)


class KKT_Optimizer(common.Sensitivity_Analysis):
    def __init__(
        self,
        cfg: KKT_Config,
        tsk: scitopt.mesh.TaskConfig,
    ):
        super().__init__(cfg, tsk)
        self.recorder.add("dC", plot_type="min-max-mean-std")
        self.recorder.add("lambda_v", ylog=False) # True
        self.lambda_v = cfg.lambda_v


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
        elements_volume_design: np.ndarray,
        elements_volume_design_sum: float,
        vol_frac: float
    ):
        cfg = self.cfg
        tsk = self.tsk
        
        vol_error = np.sum(
            rho_projected[tsk.design_elements] * elements_volume_design
        ) / elements_volume_design_sum - vol_frac
        penalty = cfg.mu_p * vol_error
        self.lambda_v = cfg.lambda_decay * self.lambda_v + penalty if iter_loop > 1 else penalty
        self.lambda_v = np.clip(self.lambda_v, cfg.lambda_lower, cfg.lambda_upper)
        self.recorder.feed_data("lambda_v", self.lambda_v)
        self.recorder.feed_data("vol_error", vol_error)
        self.recorder.feed_data("dC", dC_drho_ave)
        
        # 
        kkt_moc_log_update(
            rho=rho_candidate,
            dC=dC_drho_ave,
            lambda_v=self.lambda_v, scaling_rate=scaling_rate,
            move_limit=move_limit,
            eta=eta,
            tmp_lower=tmp_lower, tmp_upper=tmp_upper,
            rho_min=cfg.rho_min, rho_max=1.0,
            percentile=percentile,
            interpolation=cfg.interpolation
        )

    
if __name__ == '__main__':
    import argparse
    from scitopt.mesh import toy_problem

    parser = argparse.ArgumentParser(
        description=''
    )

    parser.add_argument(
        '--interpolation', '-I', type=str, default="RAMP", help=''
    )
    parser.add_argument(
        '--max_iters', '-NI', type=int, default=200, help=''
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
        '--percentile_init', '-PTI', type=float, default=60, help=''
    )
    parser.add_argument(
        '--percentile_step', '-PTR', type=int, default=2, help=''
    )
    parser.add_argument(
        '--percentile', '-PT', type=float, default=90, help=''
    )
    parser.add_argument(
        '--rho_min', '-RhM', type=float, default=1e-1, help=''
    )
    parser.add_argument(
        '--E0', '-E', type=float, default=1.0, help=''
    )
    parser.add_argument(
        '--E_min', '-EM', type=float, default=1e-6, help=''
    )
    
    parser.add_argument(
        '--eta', '-ET', type=float, default=0.3, help=''
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
        '--vol_frac_step', '-VFT', type=int, default=3, help=''
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
        '--p_init', '-PI', type=float, default=1.0, help=''
    )
    parser.add_argument(
        '--p', '-P', type=float, default=3.0, help=''
    )
    parser.add_argument(
        '--p_step', '-PRT', type=int, default=3, help=''
    )
    parser.add_argument(
        '--beta_init', '-BI', type=float, default=0.1, help=''
    )
    parser.add_argument(
        '--beta', '-B', type=float, default=5.0, help=''
    )
    parser.add_argument(
        '--beta_curvature', '-BC', type=float, default=5.0, help=''
    )
    parser.add_argument(
        '--beta_step', '-BR', type=int, default=3, help=''
    )
    parser.add_argument(
        '--beta_eta', '-BE', type=float, default=0.5, help=''
    )
    parser.add_argument(
        '--mu_p', '-MUP', type=float, default=100.0, help=''
    )
    # parser.add_argument(
    #     '--mu_d', '-MUD', type=float, default=200.0, help=''
    # )
    # parser.add_argument(
    #     '--mu_i', '-MUI', type=float, default=10.0, help=''
    # )
    parser.add_argument(
        '--lambda_v', '-LV', type=float, default=1.0, help=''
    )
    parser.add_argument(
        '--lambda_decay', '-LD', type=float, default=0.95, help=''
    )
    parser.add_argument(
        '--lambda_lower', '-BSL', type=float, default=1e-2, help=''
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
        '--export_img', '-EI', type=misc.str2bool, default=False, help=''
    )
    parser.add_argument(
        '--design_dirichlet', '-DD', type=misc.str2bool, default=True, help=''
    )
    parser.add_argument(
        '--sensitivity_filter', '-SF', type=misc.str2bool, default=True, help=''
    )
    # 
    parser.add_argument(
        '--task_name', '-T', type=str, default="toy1", help=''
    )
    parser.add_argument(
        '--mesh_path', '-MP', type=str, default="plate.msh", help=''
    )
    parser.add_argument(
        '--solver_option', '-SO', type=str, default="pyamg", help=''
    )
    
    args = parser.parse_args()
    

    if args.task_name == "toy1":
        tsk = toy_problem.toy1()
    elif args.task_name == "toy1_fine":
        tsk = toy_problem.toy1_fine()
    elif args.task_name == "toy2":
        tsk = toy_problem.toy2()
    else:
        tsk = toy_problem.toy_msh(args.task_name, args.mesh_path)
    
    print("load toy problem")
    
    print("generate KKT_Config")
    cfg = KKT_Config.from_defaults(
        **vars(args)
    )

    print("optimizer")
    optimizer = KKT_Optimizer(cfg, tsk)
    print("parameterize")
    optimizer.parameterize()
    # optimizer.parameterize(preprocess=False)
    # optimizer.load_parameters()
    print("optimize")
    optimizer.optimize()
    # optimizer.optimize_fosm()