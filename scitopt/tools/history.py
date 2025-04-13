from typing import Optional, Literal
import math
import numpy as np
import matplotlib.pyplot as plt


class HistoryLogger():
    def __init__(
        self,
        name: str,
        constants: Optional[list[float]]=None,
        constant_names: Optional[list[str]]=None,
        plot_type: Literal["min-max-mean", "min-max-mean-std"] = "min-max-mean",
        ylog: bool=False
    ):
        self.data = list()
        self.name = name
        self.constants = constants
        self.constant_names = constant_names
        self.plot_type = plot_type
        self.ylog = ylog
    
    def exists(self):
        ret = True if len(self.data) > 0 else False
        return ret
    
    def add(self, data: np.ndarray | float):
        if isinstance(data, np.ndarray):
            _temp = [np.min(data), np.mean(data), np.max(data)]
            if self.plot_type == "min-max-mean-std":
                _temp.append(np.std(data))
            self.data.append(_temp)
            
        else:
            self.data.append(float(data))
        
    def print(self):
        d = self.data[-1]
        if isinstance(d, list):
            print(f"{self.name}: min={d[0]:.3f}, mean={d[1]:.3f}, max={d[2]:.3f}")
        else:
            print(f"{self.name}: {d:.3f}")


class HistoriesLogger():
    def __init__(
        self,
        dst_path: str
    ):
        self.dst_path = dst_path
        self.histories = dict()
    
    def feed_data(self, name: str, data: np.ndarray | float):
        self.histories[name].add(data)
    
    def add(
        self,
        name: str,
        constants: Optional[list[float]] = None,
        constant_names: Optional[list[str]] = None,
        plot_type: Literal["min-max-mean", "min-max-mean-std"] = "min-max-mean",
        ylog: bool = False
    ):
        hist = HistoryLogger(
            name,
            constants=constants,
            constant_names=constant_names,
            plot_type=plot_type,
            ylog=ylog
        )
        self.histories[name] = hist
    
    def print(self):
        for k in self.histories.keys():
            if self.histories[k].exists():
                self.histories[k].print()
    
        
    def export_progress(self, fname: Optional[str] = None):
        if fname is None:
            fname = "progress.jpg"
        plt.clf()
        num_graphs = len(self.histories)
        graphs_per_page = 8
        num_pages = math.ceil(num_graphs / graphs_per_page)

        for page in range(num_pages):
            page_index = "" if num_pages == 1 else str(page)
            cols = 4
            keys = list(self.histories.keys())
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
                h = self.histories[k]
                if h.exists():
                    idx = i - start
                    p = idx // cols
                    q = idx % cols
                    d = np.array(h.data)

                    if d.ndim > 1:
                        x_array = np.array(range(d[:, 0].shape[0]))
                        ax[p, q].plot(x_array, d[:, 0], marker='o', linestyle='-', label="min")
                        ax[p, q].plot(x_array, d[:, 1], marker='o', linestyle='-', label="mean")
                        ax[p, q].plot(x_array, d[:, 2], marker='o', linestyle='-', label="max")
                        if h.plot_type == "min-max-mean-std":
                            ax[p, q].fill_between(
                                x_array,
                                d[:, 1] - d[:, 3],
                                d[:, 1] + d[:, 3],
                                color="blue", alpha=0.4, label="mean ± 1σ"
                            )
                        ax[p, q].legend(["min", "mean", "max"])
                    else:
                        ax[p, q].plot(d, marker='o', linestyle='-')

                    ax[p, q].set_xlabel("Iteration")
                    ax[p, q].set_ylabel(h.name)
                    if h.ylog is True:
                        ax[p, q].set_yscale('log')
                    ax[p, q].set_title(f"{h.name} Progress")
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
