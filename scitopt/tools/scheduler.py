from typing import Callable
from typing import Optional
import math
import numpy as np
import matplotlib.pyplot as plt


def schedule_exp_slowdown(
    it, total, start=1.0, target=0.4, rate=10.0
):
    t = it / total
    if start > target:
        return target + (start - target) * np.exp(-rate * t)
    else:
        return target - (target - start) * np.exp(-rate * t)


def schedule_exp_accelerate(
    it, total: int, start=1.0, target=0.4, rate=10.0
):
    t = it / total
    if start > target:
        return target + (start - target) * (1 - np.exp(rate * (t - 1)))
    else:
        return target - (target - start) * (1 - np.exp(rate * (t - 1)))


class Scheduler():
    
    def __init__(
        self,
        name: str,
        init_value: float,
        targe_value: float,
        rate: float,
        iters_max: int,
        func: Callable = schedule_exp_slowdown
    ):
        self.name = name
        self.init_value = init_value
        self.targe_value = targe_value
        self.iters_max = iters_max
        self.rate = rate
        self.func = func
    
    @property
    def name(self):
        return self.name

    @property
    def iters_max(self):
        return self.iters_max

    def value(self, iter: int):
        
        ret = self.func(
            iter, self.iters_max, self.init_value, self.targe_value, self.targe_value
        )
        return ret


class Schedulers():
    def __init__(self, dst_path: str):
        self.scheduler_list = []
        self.dst_path = dst_path
    
    
    def values(self, iter: int):
        ret = dict()
        for sche in self.scheduler_list:
           ret[sche.name] = sche.value(iter)
        return ret

    
    def add(
        self,
        name: str,
        init_value: float,
        targe_value: float,
        rate: float,
        iters_max: int,
        func: Callable = schedule_exp_slowdown
    ):
        self.scheduler_list.append(
            Scheduler(
                name, init_value, targe_value, rate, iters_max, func
            )
        )
    
    def export(
        self,
        fname: Optional[str]=None
    ):
        schedules = dict()
        for sche in self.scheduler_list:
           schedules[sche.name] = [ sche.value(it) for it in range(sche.iters_max)]
    
        if fname is None:
            fname = "./progress.jpg"
        plt.clf()
        num_graphs = len(schedules)
        graphs_per_page = 8
        num_pages = math.ceil(num_graphs / graphs_per_page)

        for page in range(num_pages):
            page_index = "" if num_pages == 1 else str(page)
            cols = 4
            keys = list(schedules.keys())
            start = page * cols * 2  # 2 rows on each page
            end = min(start + cols * 2, len(keys))  # 8 plots maximum on each page
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
                d = np.array(h.data)

                ax[p, q].plot(d, marker='o', linestyle='-')
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
            fig.savefig(f"{self.dst_path}/{page_index}{fname}")
            plt.close("all")