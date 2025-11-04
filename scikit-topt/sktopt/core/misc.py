import os
import argparse
import re

from sktopt import tools


def find_latest_iter_file(dst_path: str):
    pattern = re.compile(r"(\d{6})-rho\.npz")
    max_iter = -1
    latest_file = None

    for fname in os.listdir(dst_path):
        match = pattern.match(fname)
        if match:
            iter_num = int(match.group(1))
            if iter_num > max_iter:
                max_iter = iter_num
                latest_file = fname

    return max_iter, os.path.join(dst_path, latest_file) \
        if latest_file else None


def str2bool(value):
    if isinstance(value, bool):
        return value
    elif value.lower() in ('true', 't'):
        return True
    elif value.lower() in ('false', 'f'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean values is expeted')


def float_or_none(x: str) -> float | None:
    if x.lower() in ("none", "null", "nan"):
        return None
    return float(x)


def add_common_arguments(
    parser: argparse.ArgumentParser
) -> argparse.ArgumentParser:
    parser.add_argument(
        '--interpolation', '-I', type=str, default="SIMP", help=''
    )
    parser.add_argument(
        '--max_iters', '-NI', type=int, default=200, help=''
    )
    parser.add_argument(
        '--filter_type', '-FT', type=str, default="helmholtz", help=''
    )
    parser.add_argument(
        '--filter_radius_init', '-FRI', type=float, default=0.01, help=''
    )
    parser.add_argument(
        '--filter_radius', '-FR', type=float, default=0.01, help=''
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
        '--vol_frac_step', '-VFT', type=float_or_none, default=None, help=''
    )
    parser.add_argument(
        '--p_init', '-PI', type=float, default=1.0, help=''
    )
    parser.add_argument(
        '--p', '-P', type=float, default=3.0, help=''
    )
    parser.add_argument(
        '--p_step', '-PRT', type=float_or_none, default=None, help=''
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
        '--beta_step', '-BR', type=float_or_none, default=None, help=''
    )
    parser.add_argument(
        '--percentile_init', '-PTI', type=float_or_none, default=None, help=''
    )
    parser.add_argument(
        '--percentile_step', '-PTR', type=float_or_none, default=None, help=''
    )
    parser.add_argument(
        '--percentile', '-PT', type=float_or_none, default=None, help=''
    )
    parser.add_argument(
        '--rho_min', '-RhM', type=float, default=1e-1, help=''
    )
    parser.add_argument(
        '--E_min_coeff', '-EM', type=float, default=1e-3, help=''
    )
    parser.add_argument(
        '--eta', '-ET', type=float, default=0.3, help=''
    )
    parser.add_argument(
        '--beta_eta', '-BE', type=float, default=0.5, help=''
    )
    parser.add_argument(
        '--lambda_lower', type=float, default=1e-4, help=''
    )
    parser.add_argument(
        '--lambda_upper', '-BSH', type=float, default=1e+2, help=''
    )
    parser.add_argument(
        '--restart', '-RS', type=str2bool, default=False, help=''
    )
    parser.add_argument(
        '--restart_from', '-RF', type=int, default=-1, help=''
    )
    parser.add_argument(
        '--task_name', '-T', type=str, default="toy1", help=''
    )
    parser.add_argument(
        '--mesh_path', '-MP', type=str, default="plate.msh", help=''
    )
    parser.add_argument(
        '--export_img', '-EI', type=str2bool, default=True, help=''
    )
    parser.add_argument(
        '--design_dirichlet', '-DD', type=str2bool, default=True, help=''
    )
    parser.add_argument(
        '--sensitivity_filter', '-SF', type=str2bool, default=True, help=''
    )
    parser.add_argument(
        '--solver_option', '-SO', type=str, default="spsolve", help=''
    )
    parser.add_argument(
        '--n_joblib', '-NJ', type=int, default=4, help=''
    )

    return parser


def args2OC_Config_dict(args) -> dict:
    # print(args, type(args))
    ret = dict()
    keyw_schedulers = [
        "filter_radius", "move_limit", "vol_frac",
        "p", "beta", "percentile",
        # "eta"
    ]
    for keyw in keyw_schedulers:
        curvature = args["beta_curvature"] if keyw == "beta" else None
        if keyw == "move_limit":
            stype = "SawtoothDecay"
        elif keyw == "beta":
            stype = "StepAccelerating"
        elif keyw == "percentile":
            if args[keyw] is None:
                stype = "None"
            else:
                stype = "Step"
        else:
            stype = "Step"

        ret[keyw] = tools.SchedulerConfig.from_defaults(
            name=keyw,
            init_value=args[f"{keyw}_init"],
            target_value=args[keyw],
            num_steps=args[f"{keyw}_step"],
            curvature=curvature,
            scheduler_type=stype
        )

    for keyloop in args.keys():
        match_kw = next(
            (kw for kw in keyw_schedulers if keyloop.startswith(kw)), None
        )
        if keyloop == "beta_eta":
            ret[keyloop] = args[keyloop]
        if match_kw is not None:
            # print(f"{keyloop} starts with {match_kw}")
            continue

        ret[keyloop] = args[keyloop]

    return ret
