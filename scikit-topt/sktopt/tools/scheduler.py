from typing import Callable
from typing import Optional
import math
import numpy as np
import matplotlib.pyplot as plt

from sktopt.tools.logconf import mylogger
logger = mylogger(__name__)


def schedule_exp_slowdown(it, total, start=1.0, target=0.4, rate=10.0):
    if total <= 0:
        raise ValueError("total must be positive")

    t = it / total
    decay = np.exp(-rate * t)
    final_decay = np.exp(-rate)

    if start > target:
        return target + \
            (start - target) * (decay - final_decay) / (1 - final_decay)
    else:
        return target - \
            (target - start) * (decay - final_decay) / (1 - final_decay)


def schedule_exp_accelerate(
    it, total: int, start=1.0, target=0.4, rate=10.0
):
    t = it / total
    if start > target:
        return target + (start - target) * (1 - np.exp(rate * (t - 1)))
    else:
        return target - (target - start) * (1 - np.exp(rate * (t - 1)))


def schedule_step(
    it: int, total: int,
    start: float = 1.0, target: float = 0.4, num_steps: int = 10,
    **args
):
    """
    Step-function scheduler where each step value is used for
    (approximately) equal number of iterations.

    Parameters
    ----------
    it : int
        Current iteration index.
    total : int
        Total number of iterations.
    start : float
        Starting value.
    target : float
        Final target value.
    num_steps : int
        Number of discrete step values (including start and target).

    Returns
    -------
    float
        Scheduled value for the given iteration.
    """
    if total <= 0:
        raise ValueError("total must be positive")
    if num_steps <= 1:
        return target

    # Determine which step this iteration belongs to
    step_length = total / num_steps
    step_index = min(int(it // step_length), num_steps - 1)

    # Linearly divide values between start and target
    alpha = step_index / (num_steps - 1)
    value = (1 - alpha) * start + alpha * target
    return value


def schedule_step_accelerating(
    it: int,
    total: int,
    start: float = 1.0,
    target: float = 0.4,
    num_steps: int = 10,
    curvature: float = 3.0,
    **args
):
    """
    Step-function scheduler with increasing step size.

    The steps get gradually larger (nonlinear interpolation),
    controlled by 'curvature'.

    Parameters
    ----------
    it : int
        Current iteration index.
    total : int
        Total number of iterations.
    start : float
        Starting value.
    target : float
        Final target value.
    num_steps : int
        Number of steps (including start and target).
    curvature : float
        Controls how quickly the steps accelerate (larger → more aggressive).

    Returns
    -------
    float
        Scheduled value for the given iteration.
    """
    if total <= 0:
        raise ValueError("total must be positive")
    if num_steps <= 1:
        return target

    # Determine current step index
    step_length = total / num_steps
    step_index = min(int(it // step_length), num_steps - 1)

    # Use exponential-like interpolation for step values
    alpha = step_index / (num_steps - 1)
    nonlinear_alpha = alpha ** curvature

    value = (1 - nonlinear_alpha) * start + nonlinear_alpha * target
    return value


def schedule_sawtooth_decay(
    it: int,
    total: int,
    start: float = 0.1,
    target: float = 0.05,
    num_steps: int = 6,
    **args
) -> float:
    """
    Sawtooth-style scheduler: value decays linearly from `start` to `target`
    in each step, and resets at each new step.

    Parameters
    ----------
    it : int
        Current iteration index.
    total : int
        Total number of iterations.
    start : float
        Value at the beginning of each sawtooth step (e.g., high move_limit).
    target : float
        Value at the end of each sawtooth step (e.g., low move_limit).
    num_steps : int
        Number of sawtooth cycles (typically same as vol_frac steps).
    **args : dict
        Extra arguments (ignored).

    Returns
    -------
    float
        Scheduled value for the current iteration.
    """
    if total <= 0 or num_steps <= 0:
        raise ValueError("total and num_steps must be positive")

    it0 = it - 1
    total0 = total
    step_size = total0 / num_steps
    step_index = int(it0 // step_size)
    local_index = it0 - step_index * step_size
    alpha = min(local_index / step_size, 1.0)

    return (1 - alpha) * start + alpha * target


class Scheduler():
    def __init__(
        self,
        name: str,
        init_value: float,
        target_value: float,
        num_steps: float,
        iters_max: int,
        curvature: Optional[float] = None,
        func: Callable = schedule_step
    ):
        self.name = name
        self.init_value = init_value
        self.target_value = target_value
        self.iters_max = iters_max
        self.num_steps = num_steps
        self.curvature = curvature
        self.func = func

    def value(self, iter: int | np.ndarray):
        if self.num_steps < 0 or iter >= self.iters_max:
            return self.target_value

        ret = self.func(
            it=iter,
            total=self.iters_max,
            start=self.init_value,
            target=self.target_value,
            num_steps=self.num_steps,
            curvature=self.curvature,
        )
        return ret


class SchedulerStep(Scheduler):
    def __init__(
        self,
        name: str,
        init_value: float,
        target_value: float,
        num_steps: float,
        iters_max: int
    ):
        super().__init__(
            name,
            init_value,
            target_value,
            num_steps,
            iters_max,
            None,
            schedule_step
        )


class SchedulerStepAccelerating(Scheduler):
    def __init__(
        self,
        name: str,
        init_value: float,
        target_value: float,
        num_steps: float,
        iters_max: int,
        curvature: float
    ):
        super().__init__(
            name,
            init_value,
            target_value,
            num_steps,
            iters_max,
            curvature,
            schedule_step_accelerating
        )


class SchedulerSawtoothDecay(Scheduler):
    def __init__(
        self,
        name: str,
        init_value: float,
        target_value: float,
        num_steps: float,
        iters_max: int,
    ):
        super().__init__(
            name,
            init_value,
            target_value,
            num_steps,
            iters_max,
            None,
            func=schedule_sawtooth_decay
        )


class Schedulers():
    def __init__(self, dst_path: str):
        self.scheduler_list = []
        self.dst_path = dst_path

    def value_on_a_scheduler(self, key: str, iter: int) -> float:
        return self.values_as_list(
            iter, [key], export_log=False
        )[0]

    def values_as_dict(self, iter: int) -> dict:
        ret = dict()
        for sche in self.scheduler_list:
            ret[sche.name] = sche.value(iter)
        return ret

    def values_as_list(
        self, iter: int, order: list[str],
        export_log: bool = True,
        precision: int = 4
    ) -> list:
        values_dic = self.values_as_dict(iter)
        ret = [
            values_dic[k] for k in order
        ]
        if export_log is True:
            for key, value in zip(order, ret):
                logger.info(f"{key} {value:.{precision}f}")
        return ret

    def add_object(
        self, s: Scheduler
    ):
        self.scheduler_list.append(s)

    def add(
        self,
        name: str,
        init_value: float,
        target_value: float,
        num_steps: float,
        iters_max: int,
        curvature: Optional[float] = None,
        func: Callable = schedule_step
    ):
        s = Scheduler(
            name, init_value, target_value, num_steps,
            iters_max, curvature, func
        )
        # print(s.name)
        self.scheduler_list.append(s)

    def export(
        self,
        fname: Optional[str] = None
    ):
        schedules = dict()
        for sche in self.scheduler_list:
            schedules[sche.name] = [
               sche.value(it) for it in range(1, sche.iters_max+1)
            ]

        if fname is None:
            fname = "progress.jpg"
        plt.clf()
        num_graphs = len(schedules)
        graphs_per_page = 8
        num_pages = math.ceil(num_graphs / graphs_per_page)

        for page in range(num_pages):
            page_index = "0" if num_pages == 1 else str(page)
            cols = 4
            keys = list(schedules.keys())
            # 2 rows on each page
            # 8 plots maximum on each page
            start = page * cols * 2
            end = min(start + cols * 2, len(keys))
            n_graphs_this_page = end - start
            rows = math.ceil(n_graphs_this_page / cols)

            fig, ax = plt.subplots(rows, cols, figsize=(16, 4 * rows))
            ax = np.atleast_2d(ax)
            if ax.ndim == 1:
                ax = np.reshape(ax, (rows, cols))

            for i in range(start, end):
                k = keys[i]
                h = schedules[k]
                idx = i - start
                p = idx // cols
                q = idx % cols

                ax[p, q].plot(h, marker='o', linestyle='-')
                ax[p, q].set_xlabel("Iteration")
                ax[p, q].set_ylabel(k)
                ax[p, q].set_title(f"{k} Progress")
                ax[p, q].grid(True)

            total_slots = rows * cols
            used_slots = end - start
            for j in range(used_slots, total_slots):
                p = j // cols
                q = j % cols
                ax[p, q].axis("off")

            fig.tight_layout()
            print(f"{self.dst_path}/schedule-{page_index}-{fname}")
            fig.savefig(f"{self.dst_path}/schedule-{page_index}-{fname}")
            plt.close("all")
