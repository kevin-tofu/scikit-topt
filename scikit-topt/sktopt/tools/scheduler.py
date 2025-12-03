from dataclasses import dataclass
from typing import Callable
from typing import Optional
from typing import Literal
import math

import numpy as np
import matplotlib.pyplot as plt

from sktopt.tools.logconf import mylogger
logger = mylogger(__name__)


def schedule_exp_slowdown(
    it: int, total: int,
    initial_value: float = 1.0, target_value: float = 0.4, rate: float = 10.0
):
    if total <= 0:
        raise ValueError("total must be positive")

    t = it / total
    decay = np.exp(-rate * t)
    final_decay = np.exp(-rate)

    if initial_value > target_value:
        return target_value + \
            (initial_value - target_value) * (decay - final_decay) / (1 - final_decay)
    else:
        return target_value - \
            (target_value - initial_value) * (decay - final_decay) / (1 - final_decay)


def schedule_exp_accelerate(
    it: int, total: int,
    initial_value: float = 1.0, target_value: float = 0.4, rate: float = 10.0
):
    t = it / total
    if initial_value > target_value:
        return target_value + (initial_value - target_value) * (1 - np.exp(rate * (t - 1)))
    else:
        return target_value - (target_value - initial_value) * (1 - np.exp(rate * (t - 1)))


def schedule_constant(
    it: int, total: int,
    target_value: float = 0.4,
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
    initial_value : float
        Starting value.
    target_value : float
        Final target_value value.
    num_steps : int
        Number of discrete step values (including initial_value and target_value).

    Returns
    -------
    float
        Scheduled value for the given iteration.
    """
    return target_value


def schedule_step(
    it: int, total: int,
    initial_value: float = 1.0, target_value: float = 0.4,
    num_steps: int = 10,
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
    initial_value : float
        Starting value.
    target_value : float
        Final target_value value.
    num_steps : int
        Number of discrete step values (including initial_value and target_value).

    Returns
    -------
    float
        Scheduled value for the given iteration.
    """
    if total <= 0:
        raise ValueError("total must be positive")
    if num_steps <= 1:
        return target_value

    # Determine which step this iteration belongs to
    step_length = total / num_steps
    step_index = min(int(it // step_length), num_steps - 1)

    # Linearly divide values between initial_value and target_value
    alpha = step_index / (num_steps - 1)
    value = (1 - alpha) * initial_value + alpha * target_value
    return value


def schedule_step_accelerating(
    it: int,
    total: int,
    initial_value: float = 1.0,
    target_value: float = 0.4,
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
    initial_value : float
        Starting value.
    target_value : float
        Final target_value value.
    num_steps : int
        Number of steps (including initial_value and target_value).
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
        return target_value

    # Determine current step index
    step_length = total / num_steps
    step_index = min(int(it // step_length), num_steps - 1)

    # Use exponential-like interpolation for step values
    alpha = step_index / (num_steps - 1)
    nonlinear_alpha = alpha ** curvature

    value = (1 - nonlinear_alpha) * initial_value + nonlinear_alpha * target_value
    return value


def schedule_step_decelerating(
    it: int,
    total: int,
    initial_value: float = 1.0,
    target_value: float = 0.4,
    num_steps: int = 10,
    curvature: float = 3.0,
    **args
):
    """
    Step-function scheduler with decreasing step size.

    The steps change rapidly at the beginning and gradually flatten out,
    providing the opposite bias to :func:`schedule_step_accelerating`.

    Parameters
    ----------
    it : int
        Current iteration index.
    total : int
        Total number of iterations.
    initial_value : float
        Starting value.
    target_value : float
        Final target_value value.
    num_steps : int
        Number of steps (including initial_value and target_value).
    curvature : float
        Controls how quickly the steps decelerate (larger → faster early change).

    Returns
    -------
    float
        Scheduled value for the given iteration.
    """
    if total <= 0:
        raise ValueError("total must be positive")
    if num_steps <= 1:
        return target_value

    step_length = total / num_steps
    step_index = min(int(it // step_length), num_steps - 1)

    alpha = step_index / (num_steps - 1)
    nonlinear_alpha = 1 - (1 - alpha) ** curvature

    value = (1 - nonlinear_alpha) * initial_value + nonlinear_alpha * target_value
    return value


def schedule_sawtooth_decay(
    it: int,
    total: int,
    initial_value: float = 0.1,
    target_value: float = 0.05,
    num_steps: int = 6,
    **args
) -> float:
    """
    Sawtooth-style scheduler: value decays linearly from `initial_value` to `target_value`
    in each step, and resets at each new step.

    Parameters
    ----------
    it : int
        Current iteration index.
    total : int
        Total number of iterations.
    initial_value : float
        Value at the beginning of each sawtooth step (e.g., high move_limit).
    target_value : float
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

    return (1 - alpha) * initial_value + alpha * target_value


_lit_schedulers = Literal[
    'Constant',
    'Step', 'StepAccelerating', 'StepDecelerating', 'SawtoothDecay',
    'None'
]


@dataclass
class SchedulerConfig:
    """
    Configuration for continuation and parameter scheduling.

    Defines how a scalar parameter (e.g., penalization ``p``, projection
    sharpness ``beta``, or volume fraction ``vol_frac``) evolves across
    optimization iterations. Supports several scheduling strategies.

    Attributes
    ----------
    name : str, optional
        Identifier for the schedule (e.g., "p", "vol_frac").
    init_value : float, optional
        Starting value of the parameter.
        - Required for Step, StepAccelerating, SawtoothDecay.
        - Ignored if Constant.
    target_value : float, optional
        Final value of the parameter.
        - Required for Step, StepAccelerating, SawtoothDecay.
        - Used as fixed value if Constant.
    num_steps : int, optional
        Number of continuation steps or cycles.
        - Step / StepAccelerating: number of increments between init and target.
        - SawtoothDecay: number of sawtooth cycles (restarts).
        - Ignored for Constant.
    iters_max : int, optional
        Maximum total number of iterations for which the schedule is defined.
        - Used in SawtoothDecay to partition iterations into cycles.
        - Typically equals the outer optimizer's max_iters.
    curvature : float, optional
        Shape parameter used in StepAccelerating / StepDecelerating.
        - Example: curvature=2.0 accelerates change near the end.
        - Ignored in other schedulers.
    scheduler_type : {"Constant", "Step", "StepAccelerating", "StepDecelerating", "SawtoothDecay"}
        Scheduling strategy:
          - **Constant**: fixed at ``target_value``.
          - **Step**: discrete continuation from ``init_value`` → ``target_value`` in
            ``num_steps`` increments.
          - **StepAccelerating**: like Step but transition rate controlled by ``curvature``.
          - **StepDecelerating**: mirror of StepAccelerating (fast early changes, slows later).
          - **SawtoothDecay**: parameter decays linearly from ``init_value`` → ``target_value``
            within each cycle (cycle length = iters_max/num_steps), then resets to
            ``init_value`` at the start of the next cycle.
    """

    name: Optional[str] = None
    init_value: Optional[float] = None
    target_value: Optional[float] = None
    num_steps: Optional[int] = None
    iters_max: Optional[int] = None
    curvature: Optional[float] = None
    scheduler_type: _lit_schedulers = "Constant"

    @classmethod
    def from_defaults(
        cls,
        name: Optional[str] = None,
        init_value: Optional[float] = None,
        target_value: Optional[float] = None,
        num_steps: Optional[int] = None,
        iters_max: Optional[int] = None,
        curvature: Optional[float] = None,
        scheduler_type: _lit_schedulers = "Constant"
    ) -> 'SchedulerConfig':
        """
        Construct a :class:`SchedulerConfig` with validated defaults.

        This helper ensures that each scheduler type receives the proper
        parameters and fills in sensible defaults where possible.

        Parameters
        ----------
        name : str, optional
            Identifier for the schedule (e.g., "p", "vol_frac").
        init_value : float, optional
            Starting value of the parameter.
            - Required for Step, StepAccelerating, SawtoothDecay.
            - Ignored if Constant.
        target_value : float, optional
            Final value of the parameter.
            - Required for Step, StepAccelerating, SawtoothDecay.
            - Used as fixed value if Constant.
        num_steps : int, optional
            Number of continuation steps or cycles.
            - Step / StepAccelerating: number of increments from init → target.
            - SawtoothDecay: number of sawtooth cycles.
            - Ignored if Constant.
        iters_max : int, optional
            Maximum number of iterations over which the schedule is defined.
            - Used in SawtoothDecay to split iterations into cycles.
            - Ignored otherwise.
        curvature : float, optional
            Shape parameter for StepAccelerating (default ≈ 2.0).
            Controls acceleration of the step transition.
            Ignored in other schedulers.
        scheduler_type : {"Constant", "Step", "StepAccelerating", "SawtoothDecay"}, default="Constant"
            Which scheduling strategy to use:
              - **Constant**: fixed value at ``target_value``.
              - **Step**: discrete continuation from ``init_value`` → ``target_value``
                over ``num_steps`` stages.
              - **StepAccelerating**: like Step, but interpolation biased by ``curvature``.
              - **SawtoothDecay**: linear decay from ``init_value`` → ``target_value``
                within each cycle, resetting each time; ``iters_max`` defines total length.

        Returns
        -------
        SchedulerConfig
            A validated configuration object ready to be used by the scheduler
            functions.

        Raises
        ------
        ValueError
            If required parameters are missing or inconsistent with the chosen
            ``scheduler_type``.

        Notes
        -----
        - Iteration indices are assumed to be 1-based (``it=1`` is the first).
        - For Constant, ``init_value`` is overridden by ``target_value``.
        - Defaults may be auto-filled (e.g., curvature=2.0 if omitted).
        """

        # !! should add Exception Handling
        if scheduler_type == "Constant":
            if init_value is None:
                init_value = target_value
            if target_value is None:
                raise ValueError("Should set target_value")
        elif scheduler_type == "Step":
            if init_value is None:
                raise ValueError("Should set init_value")
            if target_value is None:
                raise ValueError("Should set target_value")
            if num_steps is None:
                raise ValueError("Should set num_steps")
        elif scheduler_type == "StepAccelerating":
            if init_value is None:
                raise ValueError("Should set init_value")
            if target_value is None:
                raise ValueError("Should set target_value")
            if num_steps is None:
                raise ValueError("Should set num_steps")
            if curvature is None:
                raise ValueError("Should set curvature")
        elif scheduler_type == "StepDecelerating":
            if init_value is None:
                raise ValueError("Should set init_value")
            if target_value is None:
                raise ValueError("Should set target_value")
            if num_steps is None:
                raise ValueError("Should set num_steps")
            if curvature is None:
                raise ValueError("Should set curvature")
        elif scheduler_type == "SawtoothDecay":
            if init_value is None:
                raise ValueError("Should set init_value")
            if target_value is None:
                raise ValueError("Should set target_value")
            if num_steps is None:
                raise ValueError("Should set num_steps")
        elif scheduler_type == "None":
            pass
        else:
            raise ValueError("")

        return cls(
            name=name,
            init_value=init_value,
            target_value=target_value,
            num_steps=num_steps,
            iters_max=iters_max,
            curvature=curvature,
            scheduler_type=scheduler_type,
        )

    @classmethod
    def none(
        cls,
    ) -> "SchedulerConfig":
        return cls.from_defaults(scheduler_type="None")

    @classmethod
    def constant(
        cls,
        name: Optional[str] = None,
        target_value: float = 1.0,
    ) -> "SchedulerConfig":
        """Factory for a Constant scheduler (always returns the same value)."""
        return cls.from_defaults(
            name=name,
            init_value=target_value,
            target_value=target_value,
            num_steps=None,
            iters_max=None,
            curvature=None,
            scheduler_type="Constant",
        )

    @classmethod
    def step(
        cls,
        name: Optional[str] = None,
        init_value: Optional[float] = None,
        target_value: Optional[float] = None,
        num_steps: Optional[int] = None,
        iters_max: Optional[int] = None,
    ) -> "SchedulerConfig":
        """Factory for a Step scheduler (discrete continuation)."""
        return cls.from_defaults(
            name=name,
            init_value=init_value,
            target_value=target_value,
            num_steps=num_steps,
            iters_max=iters_max,
            curvature=None,
            scheduler_type="Step",
        )

    @classmethod
    def step_accelerating(
        cls,
        name: Optional[str] = None,
        init_value: Optional[float] = None,
        target_value: Optional[float] = None,
        num_steps: Optional[int] = None,
        iters_max: Optional[int] = None,
        curvature=None
    ) -> "SchedulerConfig":
        """Factory for a StepAccelerating scheduler (nonlinear continuation with curvature)."""
        return cls.from_defaults(
            name=name,
            init_value=init_value,
            target_value=target_value,
            num_steps=num_steps,
            iters_max=iters_max,
            curvature=curvature,
            scheduler_type="StepAccelerating",
        )

    @classmethod
    def step_decelerating(
        cls,
        name: Optional[str] = None,
        init_value: Optional[float] = None,
        target_value: Optional[float] = None,
        num_steps: Optional[int] = None,
        iters_max: Optional[int] = None,
        curvature=None
    ) -> "SchedulerConfig":
        """Factory for a StepDecelerating scheduler (nonlinear continuation that slows down)."""
        return cls.from_defaults(
            name=name,
            init_value=init_value,
            target_value=target_value,
            num_steps=num_steps,
            iters_max=iters_max,
            curvature=curvature,
            scheduler_type="StepDecelerating",
        )

    @classmethod
    def sawtooth_decay(
        cls,
        name: Optional[str] = None,
        init_value: float = 0.1,
        target_value: float = 0.05,
        iters_max: int = 100,
        num_steps: int = 6,
    ) -> "SchedulerConfig":
        """
        Factory for a SawtoothDecay scheduler.

        Parameter decays linearly from init_value to target_value
        within each cycle, then resets. Total iterations = iters_max,
        cycles = num_steps.
        """
        return cls(
            name=name,
            init_value=init_value,
            target_value=target_value,
            num_steps=num_steps,
            iters_max=iters_max,
            curvature=None,
            scheduler_type="SawtoothDecay",
        )


class Scheduler():
    """
    Scheduler for **discrete continuation** from an initial value to a target value.

    This scheduler divides the iteration range into a fixed number of
    equally-sized steps, and linearly interpolates the parameter value across
    these steps. Each step produces a constant output, resulting in a staircase-
    like transition.

    Examples
    --------
    >>> cfg = SchedulerConfig.step(
    ...     name="p",
    ...     init_value=1.0,
    ...     target_value=3.0,
    ...     num_steps=5,
    ...     iters_max=100
    ... )
    >>> scheduler = SchedulerStep("p", 1.0, 3.0, 5, 100)
    >>> scheduler.value(1)
    1.0
    >>> scheduler.value(50)
    2.0

    Notes
    -----
    - This scheduler is appropriate for SIMP continuation (e.g., p = 1 → 3 → 4).
    - The step index grows proportionally to the iteration count.
    - Steps are evenly spaced unless `iters_max` is very small.

    Parameters
    ----------
    name : str
        Identifier for the scheduled parameter (e.g., ``"p"`` or ``"beta"``).
    init_value : float
        Starting value at the first step.
    target_value : float
        Final value at the last step.
    num_steps : int
        Number of discrete steps (including both endpoints).
    iters_max : int
        Total number of iterations used to map iteration → step.
    """
    def __init__(
        self,
        name: str,
        init_value: float | None,
        target_value: float | None,
        num_steps: Optional[float] = None,
        iters_max: Optional[int] = None,
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

    @classmethod
    def from_config(cls, cfg: SchedulerConfig):
            
        # 'Step', 'StepAccelerating', 'SawtoothDecay'
        if cfg.scheduler_type == 'Constant':
            func = schedule_constant
            cfg.init_value = cfg.target_value
            cfg.iters_max = None
            cfg.num_steps = None
            cfg.curvature = None
        elif cfg.scheduler_type == 'Step':
            func = schedule_step
        elif cfg.scheduler_type == 'StepAccelerating':
            func = schedule_step_accelerating
        elif cfg.scheduler_type == 'StepDecelerating':
            func = schedule_step_decelerating
        elif cfg.scheduler_type == 'SawtoothDecay':
            func = schedule_sawtooth_decay
        elif cfg.scheduler_type == 'None':
            func = None
        else:
            options = [
                'Constant',
                'Step', 'StepAccelerating', 'StepDecelerating', 'SawtoothDecay',
                'None'
            ]
            raise ValueError(
                f"{cfg.scheduler_type} not in {options}"
            )

        return cls(
            cfg.name,
            cfg.init_value,
            cfg.target_value,
            cfg.num_steps,
            iters_max=cfg.iters_max,
            curvature=cfg.curvature,
            func=func
        )

    def value(self, iter: int | np.ndarray):

        if self.target_value is None:
            return None

        if isinstance(self.num_steps, (int, float)):
            if self.num_steps < 0:
                return self.target_value
        elif self.num_steps is None:
            return self.target_value
        if iter >= self.iters_max:
            return self.target_value

        ret = self.func(
            it=iter,
            total=self.iters_max,
            initial_value=self.init_value,
            target_value=self.target_value,
            num_steps=self.num_steps,
            curvature=self.curvature,
        )
        return ret


class SchedulerStep(Scheduler):
    """
    Step-based continuation scheduler.

    This scheduler implements a *discrete continuation* strategy in which
    the parameter value (e.g., SIMP exponent ``p`` or projection sharpness
    ``beta``) transitions from ``init_value`` to ``target_value`` across
    ``num_steps`` equally sized stages. Each stage is active for
    approximately ``iters_max / num_steps`` iterations.

    Unlike gradual exponential schedules, this method produces distinct
    “plateaus” in which the parameter remains constant for a block of
    iterations before jumping to the next value. This is one of the common
    continuation techniques used in density-based topology optimization.

    Parameters
    ----------
    name : str
        Identifier for the parameter being scheduled (e.g., ``"p"`` or ``"vol_frac"``).
    init_value : float
        Starting value of the scheduled parameter.
    target_value : float
        Final value of the scheduled parameter.
    num_steps : int
        Number of continuation steps (including both endpoints).
    iters_max : int
        Total number of iterations for the optimization process.

    Notes
    -----
    - The actual scheduling logic is delegated to
      :func:`schedule_step`.
    - The value transitions linearly in ``num_steps`` discrete increments.

    Examples
    --------
    >>> cfg = SchedulerConfig.step(
    ...     name="p", init_value=1.0, target_value=3.0, num_steps=5, iters_max=100
    ... )
    >>> sched = Scheduler.from_config(cfg)
    >>> sched.value(1)
    1.0
    >>> sched.value(50)
    2.0
    >>> sched.value(100)
    3.0
    """

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
            iters_max=iters_max,
            curvature=None,
            func=schedule_step
        )


class SchedulerStepAccelerating(Scheduler):
    """
    Nonlinear step-based continuation scheduler with acceleration.

    This scheduler extends :class:`SchedulerStep` by introducing a
    *curvature-controlled nonlinear interpolation*. Earlier stages change
    slowly, while later stages accelerate more aggressively toward
    ``target_value`` as determined by the ``curvature`` parameter.

    Compared to the uniform Step schedule, this variant is useful when
    early iterations require stability (small changes) and later iterations
    benefit from rapid parameter adjustments.

    Parameters
    ----------
    name : str
        Identifier for the parameter being scheduled.
    init_value : float
        Starting value of the scheduled parameter.
    target_value : float
        Final value of the scheduled parameter.
    num_steps : int
        Number of discrete continuation steps.
    iters_max : int, optional
        Total number of optimization iterations.
    curvature : float, optional
        Nonlinearity exponent controlling acceleration.
        Higher values → stronger acceleration near the final iterations.

    Notes
    -----
    - Scheduling logic is delegated to :func:`schedule_step_accelerating`.
    - Behaves similarly to a “power-law continuation”.
    - Typically configured via :meth:`SchedulerConfig.step_accelerating`.

    Examples
    --------
    >>> cfg = SchedulerConfig.step_accelerating(
    ...     name="beta", init_value=1.0, target_value=8.0,
    ...     num_steps=6, iters_max=120, curvature=3.0
    ... )
    >>> sched = Scheduler.from_config(cfg)
    >>> sched.value(1)
    1.0
    >>> sched.value(100)  # accelerated toward the target
    7.2
    """

    def __init__(
        self,
        name: str,
        init_value: float,
        target_value: float,
        num_steps: float,
        iters_max: Optional[int] = None,
        curvature: Optional[float] = None,
    ):
        super().__init__(
            name,
            init_value,
            target_value,
            num_steps,
            iters_max=iters_max,
            curvature=curvature,
            func=schedule_step_accelerating
        )


class SchedulerStepDecelerating(Scheduler):
    """
    Nonlinear step-based continuation scheduler with deceleration.

    This is the mirror of :class:`SchedulerStepAccelerating`: the schedule
    changes quickly at the beginning and gradually flattens as iterations
    progress. Useful when you want aggressive early updates that stabilize
    over time.
    """

    def __init__(
        self,
        name: str,
        init_value: float,
        target_value: float,
        num_steps: float,
        iters_max: Optional[int] = None,
        curvature: Optional[float] = None,
    ):
        super().__init__(
            name,
            init_value,
            target_value,
            num_steps,
            iters_max=iters_max,
            curvature=curvature,
            func=schedule_step_decelerating
        )


class SchedulerSawtoothDecay(Scheduler):
    """
    Sawtooth-style cyclic decay scheduler.

    This scheduler repeatedly decays a parameter from ``init_value`` to
    ``target_value`` over each cycle, then resets sharply to ``init_value``.
    The full optimization window of ``iters_max`` iterations is divided into
    ``num_steps`` cycles.

    This pattern is useful for:
    - oscillatory move-limit control,
    - alternating aggressive and conservative update phases,
    - maintaining exploration ability in early cycles while enforcing
      stability in later cycles.

    Parameters
    ----------
    name : str
        Name of the scheduled parameter (e.g., ``"move_limit"``).
    init_value : float
        Starting value at the beginning of each cycle.
    target_value : float
        Final (lowest) value reached at the end of each cycle.
    num_steps : int
        Number of decay cycles (sawtooth teeth).
    iters_max : int, optional
        Total iterations across which cycles are distributed.

    Notes
    -----
    - Scheduling logic is delegated to :func:`schedule_sawtooth_decay`.
    - Commonly used for move-limit modulation in OC/MOC updates.
    - The value evolves linearly within each cycle.

    Examples
    --------
    >>> cfg = SchedulerConfig.sawtooth_decay(
    ...     name="move", init_value=0.3, target_value=0.1,
    ...     iters_max=120, num_steps=4
    ... )
    >>> sched = Scheduler.from_config(cfg)
    >>> sched.value(1)
    0.3
    >>> sched.value(30)
    0.2
    >>> sched.value(60)  # end of second cycle
    0.1
    >>> sched.value(61)  # reset at cycle 3
    0.3
    """

    def __init__(
        self,
        name: str,
        init_value: float,
        target_value: float,
        num_steps: float,
        iters_max: Optional[int] = None,
    ):
        super().__init__(
            name,
            init_value,
            target_value,
            num_steps,
            iters_max=iters_max,
            curvature=None,
            func=schedule_sawtooth_decay
        )


class Schedulers():
    def __init__(self, dst_path: str):
        self.scheduler_list = []
        self.dst_path = dst_path

    def set_iters_max(self, iters_max: int):
        for loop in self.scheduler_list:
            loop.iters_max = iters_max

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
                value_precision = f"{value:.{precision}f}" \
                    if value is not None else "None"
                logger.info(f"{key} {value_precision}")

        return ret

    def add_object(
        self, s: Scheduler
    ):
        self.scheduler_list.append(s)

    def add_object_from_config(
        self,
        cfg: SchedulerConfig,
        rewrite_name: str | None
    ):
        if isinstance(rewrite_name, str):
            cfg.name = rewrite_name

        self.add_object(
            Scheduler.from_config(cfg)
        )

    def add(
        self,
        name: str,
        init_value: float,
        target_value: float,
        num_steps: float,
        iters_max: Optional[int] = None,
        curvature: Optional[float] = None,
        func: Callable = schedule_step
    ):
        s = Scheduler(
            name, init_value, target_value, num_steps,
            iters_max, curvature=curvature, func=func
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
            initial_value = page * cols * 2
            end = min(initial_value + cols * 2, len(keys))
            n_graphs_this_page = end - initial_value
            rows = math.ceil(n_graphs_this_page / cols)

            fig, ax = plt.subplots(rows, cols, figsize=(16, 4 * rows))
            ax = np.atleast_2d(ax)
            if ax.ndim == 1:
                ax = np.reshape(ax, (rows, cols))

            for i in range(initial_value, end):
                k = keys[i]
                h = schedules[k]
                idx = i - initial_value
                p = idx // cols
                q = idx % cols

                ax[p, q].plot(h, marker='o', linestyle='-')
                ax[p, q].set_xlabel("Iteration")
                ax[p, q].set_ylabel(k)
                ax[p, q].set_title(f"{k} Progress")
                ax[p, q].grid(True)

            total_slots = rows * cols
            used_slots = end - initial_value
            for j in range(used_slots, total_slots):
                p = j // cols
                q = j % cols
                ax[p, q].axis("off")

            fig.tight_layout()
            print(f"{self.dst_path}/schedule-{page_index}-{fname}")
            fig.savefig(f"{self.dst_path}/schedule-{page_index}-{fname}")
            plt.close("all")
