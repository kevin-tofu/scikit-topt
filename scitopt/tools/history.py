from typing import Optional, Literal
import math
import numpy as np
import matplotlib.pyplot as plt
from scitopt.tools.logconf import mylogger
logger = mylogger(__name__)


class HistoryLogger():
    def __init__(
        self,
        name: str,
        constants: Optional[list[float]] = None,
        constant_names: Optional[list[str]] = None,
        plot_type: Literal[
            "min-max-mean", "min-max-mean-std"
        ] = "min-max-mean",
        ylog: bool = False
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
            if data.shape == ():
                self.data.append(float(data))
            else:
                _temp = [np.min(data), np.mean(data), np.max(data)]
                if self.plot_type == "min-max-mean-std":
                    _temp.append(np.std(data))
                self.data.append(_temp)
        else:
            self.data.append(float(data))

    def print(self):
        d = self.data[-1]
        if isinstance(d, list):
            logger.info(
                f"{self.name}: min={d[0]:.3f}, mean={d[1]:.3f}, max={d[2]:.3f}"
            )
        else:
            logger.info(f"{self.name}: {d:.3f}")

    def data_to_array(
        self
    ) -> tuple[np.ndarray, Optional[list[str]]]:
        if len(self.data) == 0:
            return np.array([])
        if isinstance(self.data[0], list):
            data = np.array(self.data)
            if self.plot_type == "min-max-mean-std":
                data = np.transpose(data)
                data = np.vstack((data[0], data[1], data[2], data[3]))
            else:
                data = np.transpose(data)
                data = np.vstack((data[0], data[1], data[2]))
        else:
            data = np.array(self.data)

        header = None
        if self.constants is not None:
            header = [self.name]
            if self.constant_names is not None:
                header += self.constant_names
            else:
                header += [f"constant_{i}" for i in range(len(self.constants))]
            data = np.vstack((header, data))
            data[0, 1:] = self.constants
        else:
            data = np.array(self.data)
        return (data, header)


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
        plot_type: Literal[
            "value", "min-max-mean", "min-max-mean-std"
        ] = "value",
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
                h = self.histories[k]
                if h.exists():
                    idx = i - start
                    p = idx // cols
                    q = idx % cols
                    d = np.array(h.data)
                    if d.ndim > 1:
                        x_array = np.array(range(d[:, 0].shape[0]))
                        ax[p, q].plot(
                            x_array, d[:, 0],
                            marker='o', linestyle='-', label="min"
                        )
                        ax[p, q].plot(
                            x_array, d[:, 1],
                            marker='o', linestyle='-', label="mean"
                        )
                        ax[p, q].plot(
                            x_array, d[:, 2],
                            marker='o', linestyle='-', label="max"
                        )
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
                    else:
                        ax[p, q].set_yscale('linear')
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

    def histories_to_array(
        self
    ) -> tuple[dict[str, np.ndarray], Optional[list[str]]]:
        histories = dict()
        for k in self.histories.keys():
            if self.histories[k].exists():
                data, header = self.histories[k].data_to_array()
                histories[k] = data
                if header is not None:
                    histories[f"{k}_header"] = header
        return histories, header

    def export_histories(self, fname: Optional[str] = None):
        if fname is None:
            fname = "histories.npz"
        histories, header = self.histories_to_array()
        if header is not None:
            histories["header"] = header
        else:
            histories["header"] = None
        if len(histories) == 0:
            logger.warning("No histories to save.")
            return
        if self.dst_path is None:
            logger.warning("No destination path specified.")
            return
        if not isinstance(self.dst_path, str):
            logger.warning("Destination path is not a string.")
            return
        np.savez(
            f"{self.dst_path}/{fname}", **histories
        )

    # def import_histories(
    #     self,
    #     fname: Optional[str] = None
    # ):
    #     if fname is None:
    #         fname = "histories.npz"
    #     histories = np.load(f"{self.dst_path}/{fname}", allow_pickle=True)

    #     if "header" in histories:
    #         header = histories["header"]
    #         del histories["header"]
    #     else:
    #         raise ValueError(
    #             "No header found in the histories file."
    #         )
    #     if len(histories) == 0:
    #         logger.warning("No histories to load.")
    #         return
