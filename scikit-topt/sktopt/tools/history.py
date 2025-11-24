import os
from typing import Optional, Literal
import math
import numpy as np
import matplotlib.pyplot as plt
from sktopt.tools.logconf import mylogger
logger = mylogger(__name__)


class HistorySeries():
    """
    Container for logging the evolution of a single scalar quantity
    (or its summary statistics) over iterations.

    This class stores a sequence of values or aggregated statistics
    (min / mean / max / std) and provides helpers to:

    * Append new data from scalars or arrays
    * Query the logged data as a NumPy array
    * Pretty-print the latest entry
    * Export the data in a consistent array + header format

    Parameters
    ----------
    name : str
        Name of the history (e.g. ``"compliance"``, ``"volume"``).
    constants : list of float, optional
        Optional constant values associated with this history
        (e.g. target volume fraction, penalty parameters).
        Currently stored but not used in the internal logic.
    constant_names : list of str, optional
        Names corresponding to ``constants``. Must have the same
        length as ``constants`` if both are provided.
    plot_type : {"min-max-mean", "min-max-mean-std"}, optional
        How array-valued inputs should be aggregated when passed to
        :meth:`add`:

        * ``"min-max-mean"``:
          store ``[min(x), mean(x), max(x)]``
        * ``"min-max-mean-std"``:
          store ``[min(x), mean(x), max(x), std(x)]``

        Default is ``"min-max-mean"``.
    ylog : bool, optional
        If ``True``, the history is intended to be plotted on a
        logarithmic y-axis. Used by :meth:`HistoryCollection.export_progress`.
    data : array_like, optional
        Initial data for this history. May be a 1D/2D NumPy array or a
        Python list. Internally it is stored as a Python list.
        If omitted, the history starts empty.
    """

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
        """
        Return whether this history has at least one logged entry.

        Returns
        -------
        bool
            ``True`` if at least one value has been stored,
            ``False`` otherwise.
        """

        ret = True if len(self.data) > 0 else False
        return ret

    @property
    def data_np_array(self) -> np.ndarray:
        """
        Return the logged data as a NumPy array.

        Returns
        -------
        numpy.ndarray
            Array view of the internal data. The shape depends on how
            data has been added:

            * Scalar inputs → 1D array of shape ``(n_steps,)``
            * Aggregated lists (min/mean/max[/std]) → 2D array of shape
              ``(n_steps, n_stats)``

        Raises
        ------
        ValueError
            If no data is available or if the internal buffer cannot be
            converted to a NumPy array.
        """

        try:
            value = self.data[0]
            if isinstance(value, float):
                return np.array(self.data)
            elif isinstance(value, list) or isinstance(value, np.ndarray):
                return np.array(self.data)

        except Exception as e:
            raise ValueError(f"data not exit {e}")

    def add(self, data_input: np.ndarray | float):
        """
        Append a new data point to the history.

        Parameters
        ----------
        data_input : float or numpy.ndarray
            If a scalar (Python float or 0-dimensional array), the raw
            value is appended.

            If an array with at least one dimension:

            * For ``plot_type="min-max-mean"``:
              append ``[min(x), mean(x), max(x)]``
            * For ``plot_type="min-max-mean-std"``:
              append ``[min(x), mean(x), max(x), std(x)]``
        """

        if isinstance(data_input, np.ndarray):
            if data_input.shape == ():
                self.data.append(float(data_input))
            else:
                _temp = [
                    np.min(data_input), np.mean(data_input), np.max(data_input)
                ]
                if self.plot_type == "min-max-mean-std":
                    _temp.append(np.std(data_input))
                self.data.append(_temp)
        else:
            self.data.append(float(data_input))

    def print(self):
        """
        Log the latest entry using the global ``logger``.

        If the history stores aggregated statistics, prints them as
        ``min``, ``mean``, ``max``. Otherwise prints the scalar value.
        """

        d = self.data[-1]
        if isinstance(d, list):
            logger.info(
                f"{self.name}: min={d[0]:.8f}, mean={d[1]:.8f}, max={d[2]:.8f}"
            )
        else:
            logger.info(f"{self.name}: {d:.8f}")

    def data_to_array(self) -> tuple[np.ndarray, list[str]]:
        """
        Convert the internal data to a NumPy array and a header.

        This helper is mainly used for exporting/importing histories.

        Returns
        -------
        data : numpy.ndarray
            Logged data. For aggregated histories, this has shape
            ``(n_stats, n_steps)`` where ``n_stats`` is 3 or 4
            depending on ``plot_type``. For scalar histories, this is
            a 1D array.
        header : list of str
            Metadata for this history, currently:

            * ``header[0]`` : ``name``
            * ``header[1]`` : ``plot_type``
        """

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

    def latest(self) -> float | np.ndarray:
        """
        Return the latest logged entry.

        Returns
        -------
        float or numpy.ndarray
            The most recent value. For scalar histories, this is a float.
            For aggregated histories (e.g. min/mean/max[/std]), this is
            a 1D NumPy array of statistics.

        Raises
        ------
        ValueError
            If the history is empty.
        """
        if not self.exists():
            raise ValueError(f"HistorySeries '{self.name}' has no data.")

        d = self.data[-1]
        if isinstance(d, list) or isinstance(d, np.ndarray):
            return np.array(d, dtype=float)
        else:
            return float(d)


def compare_histories_data_and_plot_type(h1, h2) -> bool:
    """
    Compare two dictionaries of ``HistorySeries`` instances.

    The dictionaries are considered equal if:

    * They have exactly the same keys
    * For each key, ``plot_type`` matches
    * Their data arrays have the same shape and are numerically
      equal up to floating-point tolerance (using
      :func:`numpy.allclose` with ``equal_nan=True``)

    Parameters
    ----------
    h1, h2 : dict[str, HistorySeries]
        Dictionaries mapping history names to :class:`HistorySeries`
        instances.

    Returns
    -------
    bool
        ``True`` if both containers are compatible as described above,
        ``False`` otherwise.
    """

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

class HistoryCollection():
    """
    Manager for multiple :class:`HistorySeries` instances.

    This class provides a convenient interface to:

    * Register multiple named histories
    * Append new data to any history
    * Print the latest values for all histories
    * Export progress plots to image files
    * Export/import histories to/from a ``.npz`` file
    * Convert all histories to NumPy arrays or attribute-style objects

    Parameters
    ----------
    dst_path : str
        Destination directory where exported figures and ``.npz``
        files will be written.
    """

    def __init__(
        self,
        dst_path: str
    ):
        self.dst_path = dst_path
        self.histories = dict()

    def feed_data(self, name: str, data: np.ndarray | float):
        """
        Append a new data point to an existing history.

        Parameters
        ----------
        name : str
            Name of the history (must already exist in
            :attr:`histories`).
        data : float or numpy.ndarray
            Data to be passed to :meth:`HistorySeries.add`.
        """
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
        """
        Register a new history under the given name.

        Parameters
        ----------
        name : str
            Name of the history to create.
        constants : list of float, optional
            Optional constant values associated with this history
            (e.g. target volume fraction, penalty parameters).
            Currently stored but not used in the internal logic.
        constant_names : list of str, optional
            Names corresponding to ``constants``.
        plot_type : {"value", "min-max-mean", "min-max-mean-std"}, optional
            Type of aggregation for array inputs:

            * ``"value"``:
              store scalar values directly. Array inputs are aggregated
              as in :meth:`HistorySeries.add` for the chosen plot_type
              in :class:`HistorySeries`.
            * ``"min-max-mean"``:
              see :class:`HistorySeries`.
            * ``"min-max-mean-std"``:
              see :class:`HistorySeries`.

            Default is ``"value"``.
        ylog : bool, optional
            If ``True``, the history is intended to be plotted on a
            logarithmic y-axis in :meth:`export_progress`.
        data : list, optional
            Initial data for the history. If provided, it is passed
            directly to :class:`HistorySeries` and stored as a list.
        """

        hist = HistorySeries(
            name,
            constants=constants,
            constant_names=constant_names,
            plot_type=plot_type,
            ylog=ylog,
            data=data
        )
        self.histories[name] = hist

    def print(self):
        """
        Print the latest value for all histories that contain data.

        Uses each :class:`HistorySeries`'s :meth:`print` method and the
        global ``logger`` to emit messages.
        """
        for k in self.histories.keys():
            if self.histories[k].exists():
                self.histories[k].print()

    def as_object(self):
        """
        Return all histories as an attribute-style object.

        Each history is exposed as an attribute whose value is the
        full NumPy array returned by
        :attr:`HistorySeries.data_np_array`.

        Returns
        -------
        obj : object
            An anonymous object such that ``obj.<name>`` is the array
            corresponding to history ``<name>``.
        """

        class AttrObj:
            pass

        obj = AttrObj()
        for name, hist in self.histories.items():
            setattr(obj, name, hist.data_np_array)
        return obj

    def as_object_latest(self):
        """
        Return the latest value of all histories as an attribute-style object.

        Each attribute corresponds to a history name and stores the
        last entry of :attr:`HistorySeries.data_np_array`.

        Returns
        -------
        obj : object
            An anonymous object such that ``obj.<name>`` is the latest
            value of history ``<name>``.
        """
        class AttrObj:
            pass

        obj = AttrObj()
        for name, hist in self.histories.items():
            setattr(obj, name, hist.data_np_array[-1])
        return obj

    def export_progress(self, fname: Optional[str] = None):
        """
        Generate and save progress plots for all histories.

        Histories are plotted in a grid of subplots, with up to eight
        graphs per page (2 rows × 4 columns). If there are more than
        eight histories, multiple pages are created.

        For aggregated histories (min/mean/max[/std]), the following
        are plotted against iteration index:

        * ``min`` (line + markers)
        * ``mean`` (line + markers)
        * ``max`` (line + markers)
        * optional shaded region ``mean ± std`` if
          ``plot_type == "min-max-mean-std"``

        For scalar histories, a single curve is plotted.

        The y-axis is logarithmic if ``ylog=True`` for that history.

        Parameters
        ----------
        fname : str, optional
            Base file name of the output image(s). If multiple pages
            are required, an index is prepended (e.g. ``"0progress.jpg"``).
            Default is ``"progress.jpg"``.
        """

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
        Convert all histories into a dictionary of NumPy arrays.

        For each history ``name``, two entries may be created:

        * ``name``:
          main data array returned by :meth:`HistorySeries.data_to_array`
        * ``f"{name}_header"``:
          header array containing metadata such as name and plot_type

        Histories with no data are skipped.

        Returns
        -------
        dict[str, numpy.ndarray]
            Mapping from keys to data/header arrays, suitable for
            saving via :func:`numpy.savez`.
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
        """
        Save all histories to a ``.npz`` file and reload them.

        This method:

        1. Collects all histories via :meth:`histories_to_array`
        2. Saves them to ``dst_path / fname`` using :func:`numpy.savez`
        3. Calls :meth:`import_histories` to reload the data and
           reconstruct :class:`HistorySeries` instances

        Parameters
        ----------
        fname : str, optional
            Output file name (without path). Defaults to
            ``"histories.npz"``.
        """

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
        Load histories from a ``.npz`` file and rebuild the loggers.

        For each data array stored under key ``name``, this method:

        * Looks for a corresponding ``f"{name}_header"`` entry to
          recover the original history name and plot_type
        * Converts the array back into the internal list format used by
          :class:`HistorySeries`
        * Reuses ``constants``, ``constant_names`` and ``ylog`` from
          any existing logger with the same name (if present)
        * Populates a new :class:`HistorySeries` and replaces
          :attr:`histories` with the reconstructed dictionary

        Parameters
        ----------
        fname : str, optional
            Input file name (without path). Defaults to
            ``"histories.npz"``.
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
                new_histories[name] = HistorySeries(
                    name=name,
                    constants=constants,
                    constant_names=constant_names,
                    plot_type=plot_type,
                    ylog=ylog,
                    data=data_list
                )

        # Replace histories with updated ones
        self.histories = new_histories

    def latest(self, name: str):
        """
        Return the latest value of a specific history.

        Parameters
        ----------
        name : str
            Name of the history to query.

        Returns
        -------
        float or numpy.ndarray
            Latest value stored in the specified history.

        Raises
        ------
        KeyError
            If the history ``name`` does not exist.
        ValueError
            If the history exists but has no data.
        """
        if name not in self.histories:
            raise KeyError(f"History '{name}' not found.")

        return self.histories[name].latest()
