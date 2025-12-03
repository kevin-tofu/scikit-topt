import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, DefaultDict, Dict, Iterator, List, Optional

from sktopt.tools.logconf import mylogger

logger = mylogger(__name__)


@dataclass
class SectionStats:
    name: str
    count: int
    total: float
    avg: float
    max: float


class SectionTimer:
    """
    Lightweight helper to measure how long named sections take.

    Examples
    --------
    >>> timer = SectionTimer()
    >>> with timer.section("assemble"):
    ...     ...
    >>> with timer.section("solve"):
    ...     ...
    >>> timer.report()

    To track nested calls, enable ``hierarchical`` so section names
    are recorded with their call stack (e.g., ``outer>inner``)::

        timer = SectionTimer(hierarchical=True)
        with timer.section("outer"):
            with timer.section("inner"):
                ...
    """

    def __init__(
        self,
        clock: Optional[Callable[[], float]] = None,
        hierarchical: bool = False,
        sep: str = ">",
    ):
        self._clock: Callable[[], float] = clock or time.perf_counter
        self._records: DefaultDict[str, List[float]] = defaultdict(list)
        self._hierarchical = hierarchical
        self._sep = sep
        self._stack: List[str] = []
        self._running_means: Dict[str, float] = {}
        self._running_counts: Dict[str, int] = {}

    @contextmanager
    def section(self, name: str) -> Iterator[None]:
        if self._hierarchical and self._stack:
            full_name = self._sep.join([*self._stack, name])
        else:
            full_name = name
        self._stack.append(name)
        start = self._clock()
        try:
            yield
        finally:
            duration = self._clock() - start
            self._records[full_name].append(duration)
            self._stack.pop()

    def wrap(self, name: str):
        """
        Decorator form of :meth:`section`.

        This is convenient for quickly instrumenting functions without
        rewriting call sites::

            @timer.wrap("my_step")
            def my_step(...):
                ...
        """

        def _decorator(func):
            def _wrapper(*args, **kwargs):
                with self.section(name):
                    return func(*args, **kwargs)
            return _wrapper

        return _decorator

    def add(self, name: str, duration: float) -> None:
        self._records[name].append(duration)

    def reset(self, name: Optional[str] = None) -> None:
        if name is None:
            self._records.clear()
            self._running_means.clear()
            self._running_counts.clear()
        else:
            self._records.pop(name, None)
            self._running_means.pop(name, None)
            self._running_counts.pop(name, None)

    def stats(self) -> List[SectionStats]:
        stats: List[SectionStats] = []
        for name, values in self._records.items():
            if not values:
                continue
            total = sum(values)
            stats.append(
                SectionStats(
                    name=name,
                    count=len(values),
                    total=total,
                    avg=total / len(values),
                    max=max(values),
                )
            )
        return stats

    def _self_time_stats(self, stats: List[SectionStats]) -> List[SectionStats]:
        sep = self._sep
        totals = {s.name: s.total for s in stats}
        children: DefaultDict[str, List[str]] = defaultdict(list)
        for name in totals:
            if sep in name:
                parent = sep.join(name.split(sep)[:-1])
                children[parent].append(name)
        self_stats: List[SectionStats] = []
        for s in stats:
            child_total = sum(totals[ch] for ch in children.get(s.name, []))
            self_time = max(s.total - child_total, 0.0)
            avg = self_time / s.count if s.count else 0.0
            self_stats.append(
                SectionStats(
                    name=s.name,
                    count=s.count,
                    total=self_time,
                    avg=avg,
                    max=self_time,  # self-time max is self-time total here
                )
            )
        return self_stats

    def summary(
        self, sort_by: str = "total", descending: bool = True
    ) -> List[SectionStats]:
        key_map = {
            "total": lambda s: s.total,
            "avg": lambda s: s.avg,
            "max": lambda s: s.max,
            "count": lambda s: s.count,
            "name": lambda s: s.name,
        }
        try:
            key_func = key_map[sort_by]
        except KeyError as exc:
            raise ValueError(
                'sort_by must be one of {"total", "avg", "max", "count", "name"}'
            ) from exc

        return sorted(self.stats(), key=key_func, reverse=descending)

    def summary_self_time(
        self, sort_by: str = "total", descending: bool = True
    ) -> List[SectionStats]:
        stats = self._self_time_stats(self.stats())
        key_map = {
            "total": lambda s: s.total,
            "avg": lambda s: s.avg,
            "max": lambda s: s.max,
            "count": lambda s: s.count,
            "name": lambda s: s.name,
        }
        try:
            key_func = key_map[sort_by]
        except KeyError as exc:
            raise ValueError(
                'sort_by must be one of {"total", "avg", "max", "count", "name"}'
            ) from exc
        return sorted(stats, key=key_func, reverse=descending)

    def report(
        self,
        sort_by: str = "total",
        descending: bool = True,
        logger_instance=None,
    ) -> str:
        stats = self.summary(sort_by=sort_by, descending=descending)
        if not stats:
            message = "No timing data collected."
            (logger_instance or logger).info(message)
            return message

        lines = [
            f"{s.name}: total={s.total:.6f}s avg={s.avg:.6f}s max={s.max:.6f}s count={s.count}"
            for s in stats
        ]

        for line in lines:
            (logger_instance or logger).info(line)

        return "\n".join(lines)

    def plot_bar(
        self,
        ax=None,
        sort_by: str = "total",
        value: str = "total",
        descending: bool = True,
        color: str = "C0",
        format_nested: Optional[bool] = None,
        stacked_nested: bool = False,
        moving_average: bool = False,
        use_self_time: bool = False,
    ):
        """
        Plot timing results as a horizontal bar chart without relying on pyplot state.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Target axes. If omitted, a new Figure/Axes is created.
        sort_by : {"total", "avg", "max", "count", "name"}
            Sorting key used before plotting.
        value : {"total", "avg", "max", "count"}
            Metric plotted on the x-axis.
        descending : bool
            Sort order for ``sort_by``.
        color : str
            Bar color passed to Matplotlib.
        format_nested : bool, optional
            If ``True`` and the timer is hierarchical, indent nested section
            labels using ``sep`` for readability. Defaults to ``hierarchical``
            flag used at construction time (ignored when ``stacked_nested`` is
            ``True``).
        stacked_nested : bool
            If ``True`` and hierarchical data are present, render a stacked bar
            for every section that has children: self-time (parent minus
            sum(children)) plus one segment per direct child. Sections without
            children are drawn as regular bars alongside the stacked groups.
        moving_average : bool
            If ``True``, plot exponential-free running averages using the
            incremental mean update (no full history stored).

        Returns
        -------
        (matplotlib.figure.Figure, matplotlib.axes.Axes)
            Figure/Axes containing the plot. If ``ax`` was provided,
            its parent figure is returned.
        """

        stats = (
            self.summary_self_time(sort_by=sort_by, descending=descending)
            if use_self_time
            else self.summary(sort_by=sort_by, descending=descending)
        )
        if not stats:
            raise ValueError("No timing data to plot.")

        metric_map = {
            "total": lambda s: s.total,
            "avg": lambda s: s.avg,
            "max": lambda s: s.max,
            "count": lambda s: s.count,
        }
        if value not in metric_map:
            raise ValueError(
                'value must be one of {"total", "avg", "max", "count"}'
            )

        if stacked_nested:
            if not any(self._sep in s.name for s in stats):
                raise ValueError("stacked_nested=True requires hierarchical section names.")

            name_to_stat = {s.name: s for s in stats}
            children_map: DefaultDict[str, List[str]] = defaultdict(list)
            for s in stats:
                if self._sep in s.name:
                    parent = s.name.rsplit(self._sep, 1)[0]
                    children_map[parent].append(s.name)

            top_levels = [s.name for s in stats if self._sep not in s.name]
            fig = None
            if ax is None:
                from matplotlib.figure import Figure

                # Extra-wide figure to clearly show stacked child segments.
                fig = Figure(figsize=(24, 0.9 * max(1, len(top_levels)) + 2))
                ax = fig.add_subplot(111)
                fig.subplots_adjust(left=0.2, right=0.95)

            import itertools
            from matplotlib import colors as mcolors

            # Fix parent/child colors to clearly separate segments.
            parent_color = "#444444"  # dark gray for parent self time
            child_palette = [
                "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                "#8c564b", "#e377c2", "#bcbd22", "#17becf", "#7f7f7f",
                "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
                "#c49c94", "#f7b6d2", "#dbdb8d", "#9edae5",
            ]
            child_colors_map = {}
            seen_labels = set()

            for y_idx, name in enumerate(top_levels):
                p_stat = name_to_stat[name]
                p_val = metric_map[value](p_stat)
                child_names = children_map.get(name, [])
                child_vals = [
                    metric_map[value](name_to_stat[ch])
                    for ch in child_names
                ]
                child_sum = sum(child_vals)
                self_val = max(p_val - child_sum, 0.0)

                left = 0.0
                lbl_self = f"{name} (self)"
                ax.barh(
                    y_idx,
                    self_val,
                    left=left,
                    color=parent_color,
                    edgecolor="black",
                    linewidth=0.5,
                    label=None if lbl_self in seen_labels else lbl_self,
                )
                seen_labels.add(lbl_self)
                left += self_val

                for ch_name, ch_val in zip(child_names, child_vals):
                    ch_label = ch_name.split(self._sep)[-1]
                    lbl = f"{name}>{ch_label}"
                    if ch_label not in child_colors_map:
                        child_colors_map[ch_label] = child_palette[len(child_colors_map) % len(child_palette)]
                    c = child_colors_map[ch_label]
                    ax.barh(
                        y_idx,
                        ch_val,
                        left=left,
                        color=c,
                        edgecolor="black",
                        linewidth=0.5,
                        label=None if lbl in seen_labels else lbl,
                    )
                    seen_labels.add(lbl)
                    left += ch_val

            ax.set_yticks(range(len(top_levels)))
            ax.set_yticklabels(top_levels)
            ax.set_xlabel(f"{value} [s]" if value in ("total", "avg", "max") else value)
            ax.set_title("Section timing (stacked by parent)")
            ax.invert_yaxis()
            handles, labels = ax.get_legend_handles_labels()
            uniq = dict(zip(labels, handles))
            ax.legend(
                uniq.values(),
                uniq.keys(),
                bbox_to_anchor=(1.04, 1),
                loc="upper left",
            )
            return (fig or ax.figure), ax

        names = [s.name for s in stats]
        fmt_nested = self._hierarchical if format_nested is None else format_nested
        if fmt_nested:
            indent = "  "
            names = [
                f"{indent * name.count(self._sep)}{name.split(self._sep)[-1]}"
                for name in names
            ]
        data = [metric_map[value](s) for s in stats]
        if moving_average:
            data = self._update_running_average(names, data)

        fig = None
        if ax is None:
            # Use Figure/Axes directly to avoid pyplot-global state.
            from matplotlib.figure import Figure

            fig = Figure(figsize=(6, 0.4 * len(stats) + 1))
            ax = fig.add_subplot(111)

        ax.barh(names, data, color=color)
        ax.set_xlabel(f"{value} [s]" if value in ("total", "avg", "max") else value)
        ax.set_title("Section timing")
        ax.invert_yaxis()
        return (fig or ax.figure), ax

    def plot_pie(
        self,
        ax=None,
        sort_by: str = "total",
        value: str = "total",
        descending: bool = True,
        colors: Optional[List[str]] = None,
        autopct: str = "%.1f%%",
        label_threshold: float = 0.05,
        min_pct_to_label: float = 1.0,
        show_legend: bool = True,
        legend_kwargs: Optional[dict] = None,
        show_total: bool = True,
        moving_average: bool = False,
        use_self_time: bool = False,
    ):
        """
        Plot timing results as a pie chart to show relative time share.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Target axes. If omitted, a new Figure/Axes is created.
        sort_by : {"total", "avg", "max", "count", "name"}
            Sorting key used before plotting.
        value : {"total", "avg", "max", "count"}
            Metric used to size the wedges.
        descending : bool
            Sort order for ``sort_by``.
        colors : list of str, optional
            Colors passed to Matplotlib ``pie``.
        autopct : str
            ``autopct`` string passed to Matplotlib ``pie``.
        label_threshold : float
            Minimum fraction (0-1) required to draw a text label on the wedge.
            Smaller slices omit the label to reduce clutter.
        min_pct_to_label : float
            Minimum percent value to render ``autopct`` text. Use ``None`` to
            always show.
        show_legend : bool
            If ``True``, draw a legend with all section names.
        legend_kwargs : dict, optional
            Extra kwargs forwarded to ``Axes.legend`` when ``show_legend`` is
            ``True``.
        show_total : bool
            If ``True``, append total runtime text to the title.
        moving_average : bool
            If ``True``, plot exponential-free running averages using the
            incremental mean update (no full history stored).

        Returns
        -------
        (matplotlib.figure.Figure, matplotlib.axes.Axes)
            Figure/Axes containing the plot. If ``ax`` was provided,
            its parent figure is returned.
        """
        stats = (
            self.summary_self_time(sort_by=sort_by, descending=descending)
            if use_self_time
            else self.summary(sort_by=sort_by, descending=descending)
        )
        if not stats:
            raise ValueError("No timing data to plot.")

        metric_map = {
            "total": lambda s: s.total,
            "avg": lambda s: s.avg,
            "max": lambda s: s.max,
            "count": lambda s: s.count,
        }
        if value not in metric_map:
            raise ValueError(
                'value must be one of {"total", "avg", "max", "count"}'
            )

        names = [s.name for s in stats]
        data = [metric_map[value](s) for s in stats]
        if moving_average:
            data = self._update_running_average(names, data)
        if all(v == 0 for v in data):
            raise ValueError("All timing values are zero; nothing to plot.")
        total = sum(data)
        unit = "s" if value in ("total", "avg", "max") else ""
        legend_labels = [
            f"{n} ({val:.3f}{unit})" if unit else f"{n} ({val})"
            for n, val in zip(names, data)
        ]
        value_label = {"total": "total", "avg": "avg", "max": "max", "count": "count"}.get(value, value)

        fig = None
        if ax is None:
            from matplotlib.figure import Figure

            fig = Figure(figsize=(12, 6))
            ax = fig.add_subplot(111)

        labels: List[str] = []
        for name, val in zip(names, data):
            frac = val / total if total else 0.0
            labels.append(name if label_threshold is None or frac >= label_threshold else "")

        def _autopct(pct: float) -> str:
            if min_pct_to_label is None or pct >= min_pct_to_label:
                return autopct % pct
            return ""

        wedges, texts, autotexts = ax.pie(
            data,
            labels=labels,
            colors=colors,
            autopct=_autopct,
            startangle=90,
            labeldistance=1.1,
            pctdistance=0.7,
        )
        ax.axis("equal")
        if show_total:
            suffix = f"(total {total:.3f}{unit})" if unit else f"(total {total:.0f})"
        else:
            suffix = ""
        ax.set_title(
            f"Section timing share {suffix}\n"
            f"(values = {value_label} self-time in {unit or 'count'})"
        )
        if show_legend:
            legend_opts = {
                "bbox_to_anchor": (1.05, 0.5),
                "loc": "center left",
                "title": "Sections",
            }
            if legend_kwargs:
                legend_opts.update(legend_kwargs)
            ax.legend(wedges, legend_labels, **legend_opts)
        return (fig or ax.figure), ax

    def plot(
        self,
        ax=None,
        sort_by: str = "total",
        value: str = "total",
        descending: bool = True,
        color: str = "C0",
        format_nested: Optional[bool] = None,
        stacked_nested: bool = False,
        kind: str = "pie",
        moving_average: bool = False,
        use_self_time: bool = False,
        **kwargs,
    ):
        """
        Plot timing results choosing between pie (default) or bar chart.

        Parameters
        ----------
        kind : {"pie", "bar"}
            Chart type. ``"pie"`` uses :meth:`plot_pie` and is the default,
            ``"bar"`` uses :meth:`plot_bar`.
        Other parameters
            Passed through to the selected plotting function.
        """
        if kind == "pie":
            return self.plot_pie(
                ax=ax,
                sort_by=sort_by,
                value=value,
                descending=descending,
                moving_average=moving_average,
                use_self_time=use_self_time,
                **kwargs,
            )
        if kind == "bar":
            return self.plot_bar(
                ax=ax,
                sort_by=sort_by,
                value=value,
                descending=descending,
                color=color,
                format_nested=format_nested,
                stacked_nested=stacked_nested,
                moving_average=moving_average,
                use_self_time=use_self_time,
                **kwargs,
            )
        raise ValueError('kind must be one of {"pie", "bar"}')

    def save_plot(
        self,
        filepath: str,
        sort_by: str = "total",
        value: str = "total",
        descending: bool = True,
        color: str = "C0",
        dpi: int = 150,
        format_nested: Optional[bool] = None,
        stacked_nested: bool = False,
        kind: str = "pie",
        use_self_time: bool = False,
        moving_average: bool = False,
        **kwargs,
    ) -> None:
        """
        Render and save the timing plot to ``filepath``.

        This helper builds its own Figure/Axes (no pyplot state), so it can be
        used safely inside loops.
        """
        stats = (
            self.summary_self_time(sort_by=sort_by, descending=descending)
            if use_self_time
            else self.summary(sort_by=sort_by, descending=descending)
        )
        if not stats:
            raise ValueError("No timing data to plot.")

        fig, _ = self.plot(
            ax=None,
            sort_by=sort_by,
            value=value,
            descending=descending,
            color=color,
            format_nested=format_nested,
            stacked_nested=stacked_nested,
            kind=kind,
            use_self_time=use_self_time,
            moving_average=moving_average,
            **kwargs,
        )
        fig.tight_layout()
        fig.savefig(filepath, dpi=dpi)
        fig.clf()

    def _update_running_average(self, names: List[str], data: List[float]) -> List[float]:
        """Incremental mean update: new_avg = old_avg + (x - old_avg) / n."""
        smoothed: List[float] = []
        for name, val in zip(names, data):
            prev_count = self._running_counts.get(name, 0) + 1
            prev_mean = self._running_means.get(name, val)
            new_mean = prev_mean + (val - prev_mean) / prev_count
            self._running_counts[name] = prev_count
            self._running_means[name] = new_mean
            smoothed.append(new_mean)
        return smoothed
