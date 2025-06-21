import os
from typing import Optional, Literal
import math
import numpy as np
import matplotlib.pyplot as plt
from sktopt.tools.logconf import mylogger
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
        ylog: bool = False,
        data: Optional[np.ndarray] = None
    ):
        self.name = name
        self.constants = constants
        self.constant_names = constant_names
        self.plot_type = plot_type
        self.ylog = ylog
        if data is not None:
            if isinstance(data, np.ndarray):
                self.data = data.tolist()
            elif isinstance(data, list):
                self.data = data
            else:
                raise TypeError("data must be a list or np.ndarray")
        else:
            self.data = []

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

    def data_to_array(self) -> tuple[np.ndarray, list[str]]:
        if len(self.data) == 0:
            return np.array([]), [self.name, self.plot_type]

        # データ本体
        if isinstance(self.data[0], list):
            data = np.array(self.data)
            if self.plot_type == "min-max-mean-std":
                data = data.T[:4]
            else:
                data = data.T[:3]
        else:
            data = np.array(self.data)

        header = [self.name, self.plot_type]
        return data, header


def compare_histories_data_and_plot_type(h1, h2) -> bool:
    if h1.keys() != h2.keys():
        return False

    for key in h1:
        a = h1[key]
        b = h2[key]

        if a.plot_type != b.plot_type:
            return False

        a_data = np.array(a.data, dtype=float)
        b_data = np.array(b.data, dtype=float)
        if a_data.shape != b_data.shape:
            return False
        if not np.allclose(a_data, b_data, equal_nan=True):
            return False

    return True


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
        ylog: bool = False,
        data: Optional[list] = None
    ):
        hist = HistoryLogger(
            name,
            constants=constants,
            constant_names=constant_names,
            plot_type=plot_type,
            ylog=ylog,
            data=data
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

    def histories_to_array(self) -> dict[str, np.ndarray]:
        """
        Converts all history loggers into a dictionary of arrays.
        Each logger contributes its main data and a header array.
        """
        histories = {}

        for name, logger in self.histories.items():
            if not logger.exists():
                continue

            data, header = logger.data_to_array()
            histories[name] = data

            if header is not None:
                histories[f"{name}_header"] = np.array(header, dtype=str)

        return histories

    def export_histories(self, fname: Optional[str] = None):
        if fname is None:
            fname = "histories.npz"

        histories = self.histories_to_array()
        data_keys = [k for k in histories.keys() if not k.endswith("_header")]

        if len(data_keys) == 0:
            logger.warning("No histories to save.")
            return

        if not isinstance(self.dst_path, str):
            logger.warning("Invalid destination path.")
            return

        path = os.path.join(self.dst_path, fname)
        np.savez(path, **histories)

        # import copy
        # before_histories = copy.deepcopy(self.histories)
        self.import_histories(fname)
        # print(
        #     compare_histories_data_and_plot_type(
        #         before_histories, self.histories
        #     )
        # )

    def import_histories(self, fname: Optional[str] = None):
        """
        Load histories from a .npz file.
        - Restores data and plot_type from file
        - Reuses existing constants, constant_names, ylog if available
        in self.histories
        - Fully replaces self.histories after loading
        """
        if fname is None:
            fname = "histories.npz"

        if not isinstance(self.dst_path, str):
            logger.warning("Invalid destination path.")
            return

        path = os.path.join(self.dst_path, fname)
        if not os.path.exists(path):
            logger.warning(f"File not found: {path}")
            return

        # Create new container to accumulate updated histories
        new_histories = {}

        with np.load(path, allow_pickle=True) as data:
            for key in data.files:
                if key.endswith("_header"):
                    continue

                arr = data[key]
                header_key = f"{key}_header"

                if header_key in data:
                    header = data[header_key].tolist()
                    name = header[0]
                    plot_type = header[1] if len(header) > 1 else "min-max-mean"
                else:
                    name = key
                    plot_type = "value"

                # Convert data to list format
                if arr.ndim == 2 and arr.shape[0] > 1:
                    data_list = [list(x) for x in arr.T]
                else:
                    data_list = arr.tolist()

                # Reuse previous information if available
                existing = self.histories.get(name)
                constants = existing.constants if existing else None
                constant_names = existing.constant_names if existing else None
                ylog = existing.ylog if existing else False

                # Add to temporary dictionary
                new_histories[name] = HistoryLogger(
                    name=name,
                    constants=constants,
                    constant_names=constant_names,
                    plot_type=plot_type,
                    ylog=ylog,
                    data=data_list
                )

        # Replace histories with updated ones
        self.histories = new_histories

