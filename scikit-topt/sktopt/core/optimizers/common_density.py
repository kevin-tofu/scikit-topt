import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from dataclasses import asdict
from typing import Literal
import inspect
import shutil
import json

import numpy as np
import sktopt
from sktopt import tools
from sktopt.core import derivatives, projection
from sktopt.core import visualization
from sktopt.mesh import visualization as visualization_mesh
from sktopt import fea
from sktopt.fea import composer
from sktopt import filters
from sktopt.core import misc
from sktopt.tools.logconf import mylogger
logger = mylogger(__name__)
import logging
logging.getLogger("skfem").setLevel(logging.WARNING)


@dataclass
class DensityMethodConfig():
    """
    Configuration for density-based topology optimization (SIMP/RAMP).

    This configuration collects the main numerical settings for
    density-based topology optimization, including interpolation models,
    continuation schedules (via :class:`~sktopt.tools.SchedulerConfig`),
    filtering/projection parameters, and solver options. It is intended
    for sensitivity-based methods such as SIMP and RAMP and is shared by
    both OC and MOC-style update rules.

    Notes on continuation
    ---------------------
    Parameters that change during optimization are represented by
    :class:`~sktopt.tools.SchedulerConfig`:

    - ``p``: Penalization power (e.g., 1.0 → 3.0).
    - ``vol_frac``: Target volume fraction (often constant, e.g. 0.8).
    - ``beta``: Heaviside / projection sharpness (e.g., 1.0 → 2.0 or higher).
    - ``filter_radius``: Filter radius (typically kept constant).

    Each scheduler defines how its value evolves during the optimization
    (e.g., :func:`SchedulerConfig.step`,
    :func:`SchedulerConfig.step_accelerating`,
    :func:`SchedulerConfig.constant`) and may include shape parameters
    such as ``curvature`` for accelerating schedules.

    Attributes
    ----------
    dst_path : str
        Output directory for results and logs.
    interpolation : {"SIMP", "RAMP"}
        Material interpolation model used for penalization.
    record_times : int
        Number of snapshots recorded during optimization.
    max_iters : int
        Maximum number of optimization iterations.

    beta_eta : float
        Additional damping parameter used in OC/MOC-style updates
        (for example, to smooth dual variables or stabilize
        multiplicative/log-space updates).
    eta : float
        Damping or learning-rate–like exponent used in the OC/MOC
        multiplicative update rule.

    p : :class:`~sktopt.tools.SchedulerConfig`
        Continuation schedule for the penalization power.
        Default: ``step(init_value=1.0, target_value=3.0, num_steps=3)``.
    vol_frac : :class:`~sktopt.tools.SchedulerConfig`
        Schedule for the target volume fraction constraint.
        Default: ``constant(target_value=0.8)``.
    beta : :class:`~sktopt.tools.SchedulerConfig`
        Continuation schedule for Heaviside / projection sharpness (β).
        Default: ``step_accelerating(init_value=1.0, target_value=2.0,
        num_steps=3, curvature=2.0)``.
    filter_type : {"spacial", "helmholtz"}
        Type of filter applied to the density field (spatial or Helmholtz).
    filter_radius : :class:`~sktopt.tools.SchedulerConfig`
        Schedule for the filter radius.
        Default: ``constant(target_value=0.01)``.

    E_min_coeff : float
        Proportional constant that defines the minimum Young’s modulus
        as ``E_min_coeff * E0`` (typically :math:`10^{-3}`).
    rho_min : float
        Minimum density clamp to avoid singular stiffness matrices.
    rho_max : float
        Maximum density clamp (typically 1.0).

    restart : bool
        If ``True``, resume optimization from a saved state in ``dst_path``.
    restart_from : int
        Iteration index to resume from. Use ``-1`` to auto-detect the
        latest checkpoint.

    export_img : bool
        If ``True``, export images of the density field during optimization.
    export_img_opaque : bool
        If ``True``, use opaque rendering for exported images.

    design_dirichlet : bool
        If ``True``, Dirichlet boundary elements are included in the
        design domain.
    sensitivity_filter : bool
        If ``True``, apply filtering directly to sensitivity fields
        in addition to density filtering.

    solver_option : {"spsolve", "cg_pyamg"}
        Linear solver for the state analysis.
        ``"cg_pyamg"`` enables multigrid-accelerated CG via PyAMG.
    scaling : bool
        If ``True``, apply length/force scaling to normalize geometry
        and loads for improved numerical conditioning.
    n_joblib : int
        Number of parallel workers used in joblib-enabled sections.

    """

    dst_path: str = "./result/pytests"
    interpolation: Literal["SIMP", "RAMP"] = "SIMP"
    record_times: int = 20
    max_iters: int = 200
    beta_eta: float = 0.50
    eta: float = 0.6
    p: tools.SchedulerConfig = field(
        default_factory=lambda: tools.SchedulerConfig.step(
            init_value=1.0, target_value=3.0, num_steps=3
        )
    )
    vol_frac: tools.SchedulerConfig = field(
        default_factory=lambda: tools.SchedulerConfig.constant(
            target_value=0.8
        )
    )
    beta: tools.SchedulerConfig = field(
        default_factory=lambda: tools.SchedulerConfig.step_accelerating(
            init_value=1.0, target_value=2.0,
            num_steps=3,
            curvature=2.0
        )
    )
    filter_type: Literal[
        "spacial", "helmholtz"
    ] = "helmholtz"
    filter_radius: tools.SchedulerConfig = field(
        default_factory=lambda: tools.SchedulerConfig.constant(
            target_value=0.01
        )
    )
    E_min_coeff: float = 1e-3
    rho_min: float = 1e-2
    rho_max: float = 1.0
    restart: bool = False
    restart_from: int = -1
    export_img: bool = False
    export_img_opaque: bool = False
    design_dirichlet: bool = False
    sensitivity_filter: bool = False
    solver_option: Literal["spsolve", "cg_pyamg"] = "spsolve"
    scaling: bool = False
    n_joblib: int = 1

    @classmethod
    def from_defaults(cls, **args) -> 'DensityMethodConfig':
        sig = inspect.signature(cls)
        valid_keys = sig.parameters.keys()
        filtered_args = {k: v for k, v in args.items() if k in valid_keys}
        return cls(**filtered_args)

    @classmethod
    def import_from(cls, path: str) -> 'DensityMethodConfig':
        import json
        with open(f"{path}/cfg.json", "r") as f:
            data = json.load(f)
        return cls(**data)

    def export(self, path: str):
        with open(f"{path}/cfg.json", "w") as f:
            json.dump(asdict(self), f, indent=2)

    # def export(self, path: str):
    #     import yaml
    #     with open(f"{path}/cfg.yaml", "w") as f:
    #         yaml.dump(asdict(self), f, sort_keys=False)

    # @classmethod
    # def import_from(cls, path: str):
    #     import yaml
    #     with open(f"{path}/cfg.yaml", "r") as f:
    #         data = yaml.safe_load(f)
    #     return OC_Config(**data)

    def vtu_path(self, iter_num: int):
        return f"{self.dst_path}/mesh_rho/info_mesh-{iter_num:08d}.vtu"

    def image_path(self, iter_num: int, prefix: str):
        if self.export_img:
            return f"{self.dst_path}/mesh_rho/info_{prefix}-{iter_num:08d}.jpg"
        else:
            return None


@dataclass
class DensityMethod_OC_Config(DensityMethodConfig):
    """
    Configuration class for OC-style density updates.

    This subclass of :class:`DensityMethodConfig` adds parameters that are
    specific to Optimality Criteria (OC) and related multiplicative update
    schemes. It controls the move-limit strategy, optional percentile-based
    sensitivity scaling, and the bracket for the Lagrange multiplier used
    in the volume-constraint bisection.

    Attributes
    ----------
    lambda_lower : float
        Lower bound for the Lagrange multiplier used in the bisection
        enforcing the volume constraint. Must be strictly positive.
    lambda_upper : float
        Upper bound for the Lagrange multiplier used in the bisection
        enforcing the volume constraint.

    percentile : :class:`~sktopt.tools.SchedulerConfig`
        Optional schedule for percentile-based scaling of sensitivity
        fields. When set to ``SchedulerConfig.none()``, percentile
        scaling is disabled and raw sensitivities are used.
    move_limit : :class:`~sktopt.tools.SchedulerConfig`
        Schedule for the OC move limit applied to density updates.
        By default, a sawtooth-decay schedule is used, e.g.
        ``sawtooth_decay("move_limit", 0.3, 0.1, 3)``, which gradually
        tightens the allowed per-iteration change in density.

    """

    lambda_lower: float = 1e-7
    lambda_upper: float = 1e+7
    percentile: tools.SchedulerConfig = field(
        default_factory=lambda: tools.SchedulerConfig.none()
    )
    move_limit: tools.SchedulerConfig = field(
        default_factory=lambda: tools.SchedulerConfig.sawtooth_decay(
            "move_limit", 0.3, 0.1, 3
        )
    )


def interpolation_funcs(cfg: DensityMethodConfig):
    if cfg.interpolation == "SIMP":
        # vol_frac = cfg.vol_frac_init
        return [
            composer.simp_interpolation, derivatives.dC_drho_simp
        ]
    elif cfg.interpolation == "RAMP":
        # vol_frac = 0.4
        return [
            composer.ramp_interpolation, derivatives.dC_drho_ramp
        ]
    else:
        raise ValueError("Interpolation method must be SIMP or RAMP.")


class DensityMethodBase(ABC):
    @abstractmethod
    def add_recorder(self):
        pass

    @abstractmethod
    def scale(self):
        pass

    @abstractmethod
    def unscale(self):
        pass

    @abstractmethod
    def init_schedulers(self, export: bool = True):
        pass

    @abstractmethod
    def parameterize(self):
        pass

    @abstractmethod
    def load_parameters(self):
        pass

    @abstractmethod
    def initialize_density(self):
        pass

    @abstractmethod
    def initialize_params(self):
        pass

    @abstractmethod
    def optimize(self):
        pass

    @abstractmethod
    def rho_update(
        self,
        iter_num: int,
        rho_design_eles: np.ndarray,
        rho_projected: np.ndarray,
        dC_drho_design_eles: np.ndarray,
        u_dofs: np.ndarray,
        energy_mean: np.ndarray,
        scaling_rate: np.ndarray,
        move_limit: float,
        eta: float,
        rho_clip_lower: np.ndarray,
        rho_clip_upper: np.ndarray,
        lambda_lower: float,
        lambda_upper: float,
        percentile: float | None,
        elements_volume_design: np.ndarray,
        elements_volume_design_sum: float,
        vol_frac: float
    ):
        pass


class DensityMethod(DensityMethodBase):
    """
    Core driver for sensitivity-based density topology optimization.

    This class implements the common workflow shared by multiple
    optimization strategies (e.g., OC, LogMOC). It handles

    - problem setup (scaling, restart, output directories),
    - finite element analysis (FEA) for elasticity / heat conduction,
    - density initialization and masking of fixed / boundary elements,
    - density and sensitivity filtering, and
    - Heaviside-type projection and sensitivity chain rules,

    while leaving the actual density update rule to subclasses via
    :meth:`rho_update`.

    Typical usage involves:

    - Assembling the global stiffness / conductivity matrix
      (via an internal :class:`~sktopt.fea.FEM_*` instance),
    - Solving the state problem under given loads and boundary conditions,
    - Computing compliance (or thermal) objectives and sensitivities,
    - Applying density / sensitivity filters (e.g. spatial, Helmholtz),
    - Projecting intermediate densities with a smooth Heaviside function, and
    - Delegating the update of design densities to a subclass-specific
      implementation (OC, MOC variants, etc.).

    Responsibilities
    ----------------
    - Manage the overall optimization loop (iterations, restart, output).
    - Initialize and update continuation schedules
      (penalization, volume fraction, projection sharpness, move limits, etc.).
    - Apply filtering and projection consistently to densities and gradients.
    - Evaluate objective values and record histories / visualization outputs.
    - Call :meth:`rho_update` to perform the actual density update
      (including, e.g., Lagrange multiplier bisection for OC).

    Notes
    -----
    This class serves as the backbone of density-based topology
    optimization in Scikit-Topt. All algorithm-specific update logic
    (e.g., OC / MOC density updates, dual variable handling) must be
    implemented in subclasses by overriding :meth:`rho_update`.

    Attributes
    ----------
    cfg : DensityMethodConfig
        Configuration object holding numerical and algorithmic settings,
        such as interpolation model, filters, continuation schedules,
        and solver options.
    tsk : sktopt.mesh.FEMDomain
        Problem description including mesh, basis, boundary conditions,
        and load vectors.
    fem : sktopt.fea.FEM_SimpLinearElasticity or \
          sktopt.fea.FEM_SimpLinearHeatConduction
        Finite element solver used for state and sensitivity evaluation,
        constructed from ``tsk`` and ``cfg``.
    schedulers : sktopt.tools.Schedulers
        Collection of :class:`~sktopt.tools.SchedulerConfig` instances
        controlling continuation of parameters such as ``p``, ``vol_frac``,
        ``beta``, ``move_limit``, ``percentile``, and ``filter_radius``.
    """

    def __init__(
        self,
        cfg: DensityMethodConfig,
        tsk: sktopt.mesh.FEMDomain,
    ):
        self.cfg = cfg
        self.tsk = tsk
        if cfg.scaling is True:
            self.scale()

        if not os.path.exists(self.cfg.dst_path):
            os.makedirs(self.cfg.dst_path)
        self.cfg.export(self.cfg.dst_path)

        if cfg.design_dirichlet is False:
            self.tsk.exlude_dirichlet_from_design()

        if cfg.restart is True:
            self.load_parameters()
        else:
            if os.path.exists(f"{self.cfg.dst_path}/mesh_rho"):
                shutil.rmtree(f"{self.cfg.dst_path}/mesh_rho")
            os.makedirs(f"{self.cfg.dst_path}/mesh_rho")
            if not os.path.exists(f"{self.cfg.dst_path}/data"):
                os.makedirs(f"{self.cfg.dst_path}/data")

        if isinstance(tsk, sktopt.mesh.LinearElasticity):
            self.fem = fea.FEM_SimpLinearElasticity(
                tsk, cfg.E_min_coeff,
                density_interpolation=interpolation_funcs(cfg)[0],
                solver_option=cfg.solver_option,
                n_joblib=cfg.n_joblib
            )
        elif isinstance(tsk, sktopt.mesh.LinearHeatConduction):
            self.fem = fea.FEM_SimpLinearHeatConduction(
                tsk, cfg.E_min_coeff,
                density_interpolation=interpolation_funcs(cfg)[0],
                solver_option=cfg.solver_option,
                n_joblib=cfg.n_joblib
            )
        else:
            raise NotImplementedError(
                ""
            )
        # self.recorder = self.add_recorder(tsk)
        self.schedulers = tools.Schedulers(self.cfg.dst_path)

    def add_recorder(
        self, tsk: sktopt.mesh.FEMDomain
    ) -> tools.HistoryCollection:
        recorder = tools.HistoryCollection(self.cfg.dst_path)
        recorder.add("rho_projected", plot_type="min-max-mean-std")
        recorder.add("energy", plot_type="min-max-mean-std")
        recorder.add("vol_error")
        # if isinstance(tsk.neumann_values, list):
        if tsk.n_tasks > 1:
            recorder.add("u_max", plot_type="min-max-mean-std")
        else:
            recorder.add("u_max")
        recorder.add("compliance", ylog=True)
        recorder.add("scaling_rate", plot_type="min-max-mean-std")
        return recorder

    def params_latest(self):
        return self.recorder.as_object_latest()

    def scale(self):
        bbox = np.ptp(self.tsk.mesh.p, axis=1)
        L_max = np.max(bbox)
        # L_mean = np.mean(bbox)
        # L_geom = np.cbrt(np.prod(bbox))
        self.L_scale = L_max
        # self.tsk.mesh /= self.L_scale
        self.F_scale = 10**5
        self.tsk.scale(
            1.0 / self.L_scale, 1.0 / self.F_scale
        )

    def unscale(self):
        self.tsk.scale(
            self.L_scale, self.F_scale
        )

    def init_schedulers(self, export: bool = True):

        cfg = self.cfg
        self.schedulers.add_object_from_config(cfg.p, "p")
        self.schedulers.add_object_from_config(cfg.vol_frac, "vol_frac")
        self.schedulers.add_object_from_config(cfg.move_limit, "move_limit")
        self.schedulers.add_object_from_config(cfg.beta, "beta")
        self.schedulers.add_object_from_config(cfg.percentile, "percentile")
        self.schedulers.add_object_from_config(
            cfg.filter_radius, "filter_radius"
        )
        # self.schedulers.add_object_from_config(cfg.eta)
        if isinstance(cfg.eta, tools.SchedulerConfig):
            self.schedulers.add_object_from_config(cfg.eta, "eta")
        else:
            # constant
            self.schedulers.add(
                "eta",
                self.cfg.eta,
                self.cfg.eta,
                -1,
                self.cfg.max_iters
            )
        self.schedulers.set_iters_max(cfg.max_iters)

        if export:
            self.schedulers.export()

    def parameterize(self):

        if self.cfg.filter_type == "spacial":
            self.filter = filters.SpacialFilter.from_defaults(
                self.tsk.mesh, self.tsk.elements_volume,
                self.cfg.filter_radius.init_value,
                design_mask=self.tsk.design_mask,
            )
        elif self.cfg.filter_type == "helmholtz":
            self.filter = filters.HelmholtzFilterNodal.from_defaults(
                self.tsk.mesh, self.tsk.elements_volume,
                self.cfg.filter_radius.init_value,
                design_mask=self.tsk.design_mask
            )
        # elif self.cfg.filter_type == "helmholtz_ele":
        #     self.filter = filters.HelmholtzFilterElement.from_defaults(
        #         self.tsk.mesh, self.tsk.elements_volume,
        #         self.cfg.filter_radius.init_value,
        #         design_mask=self.tsk.design_mask,
        #         solver_option="cg_pyamg",
        #     )
        else:
            raise ValueError("should be spacial or helmholtz")

    def load_parameters(self):
        # self.filter = filters.HelmholtzFilterElement.from_file(
        #     f"{self.cfg.dst_path}/data"
        # )
        pass

    def initialize_density(self):
        tsk = self.tsk
        cfg = self.cfg
        val_init = cfg.vol_frac.init_value \
            if cfg.vol_frac.init_value is not None else cfg.vol_frac.target_value
        rho = np.zeros_like(tsk.all_elements, dtype=np.float64)
        iter_begin = 1
        if cfg.restart is True:
            if cfg.restart_from > 0:
                data_dir = f"{cfg.dst_path}/data"
                data_fname = f"{cfg.restart_from:06d}-rho.npz"
                data_path = f"{data_dir}/{data_fname}"
                data = np.load(data_path)
                iter_begin = cfg.restart_from + 1
            else:
                iter_num, data_path = misc.find_latest_iter_file(
                    f"{cfg.dst_path}/data"
                )
                data = np.load(data_path)
                iter_begin = iter_num + 1
            iter_end = cfg.max_iters + 1
            self.recorder.import_histories()
            rho[tsk.design_elements] = data["rho_design_elements"]
            del data
        else:
            rho += val_init
            np.clip(rho, cfg.rho_min, cfg.rho_max, out=rho)
            iter_end = cfg.max_iters + 1

        if cfg.design_dirichlet is True:
            rho[tsk.neumann_elements] = 1.0
        else:
            rho[tsk.dirichlet_neumann_elements] = 1.0
        rho[tsk.fixed_elements] = 1.0
        return rho, iter_begin, iter_end

    def initialize_params(self):
        """
        Initialize parameters and preallocate work arrays for density-based topology
        optimization using the SIMP method.

        This prepares all arrays required for the iterative loop. By preallocating and
        reusing arrays in-place, repeated allocations are avoided and performance improves.

        Returns
        -------
        iter_begin : int
            Starting iteration index.
        iter_end : int
            Final iteration index.
        rho : ndarray, shape (n_elements,)
            Current element-wise design density.
        rho_prev : ndarray, shape (n_elements,)
            Copy of densities from the previous iteration.
        rho_filtered : ndarray, shape (n_elements,)
            Densities after applying the density filter (e.g., Helmholtz filter).
        rho_projected : ndarray, shape (n_elements,)
            Densities after applying the projection (e.g., Heaviside projection).
        dH_drho : ndarray, shape (n_elements,)
            Derivative of the projection with respect to the filtered densities.
        grad_filtered : ndarray, shape (n_elements,)
            Aggregated gradient after combining sensitivity with projection derivative.
        dC_drho_projected : ndarray, shape (n_elements,)
            Compliance sensitivities with respect to the projected densities.
        energy_mean : ndarray, shape (n_elements,)
            Average element strain energy over all load cases.
        dC_drho_full : ndarray, shape (n_elements,)
            Compliance sensitivities mapped back to the full element set.
        dC_drho_design_eles : ndarray, shape (n_design_elements,)
            Compliance sensitivities restricted to design elements only.
        scaling_rate : ndarray, shape (n_design_elements,)
            Scaling factors used in the update scheme (e.g., OC/MOC).
        rho_design_eles : ndarray, shape (n_design_elements,)
            Subset of densities corresponding to design elements only.
        rho_clip_lower : ndarray, shape (n_design_elements,)
            Lower clipping bounds for density updates.
        rho_clip_upper : ndarray, shape (n_design_elements,)
            Upper clipping bounds for density updates.
        u_dofs : ndarray, shape (ndof, n_load_cases)
            Displacement field solutions for each degree of freedom and each load case.
        filter_radius : float
            Initial density-filter radius, determined from a fixed value or a schedule.

        Notes
        -----
        - This setup targets density-based (SIMP) topology optimization.
        - Arrays are reused in-place across iterations to minimize allocation overhead.
        """
        tsk = self.tsk
        cfg = self.cfg
        rho, iter_begin, iter_end = self.initialize_density()
        rho_prev = np.zeros_like(rho)
        rho_filtered = np.zeros_like(rho)
        rho_projected = np.zeros_like(rho)
        dH_drho = np.empty_like(rho)
        grad_filtered = np.empty_like(rho)
        dC_drho_projected = np.empty_like(rho)
        energy_mean = np.zeros_like(rho)
        dC_drho_full = np.zeros_like(rho)
        dC_drho_design_eles = np.zeros_like(rho[tsk.design_elements])
        scaling_rate = np.empty_like(rho[tsk.design_elements])
        rho_design_eles = np.empty_like(rho[tsk.design_elements])
        rho_clip_lower = np.empty_like(rho[tsk.design_elements])
        rho_clip_upper = np.empty_like(rho[tsk.design_elements])
        # force_vec_list = tsk.neumann_linear \
        #     if isinstance(tsk.neumann_linear, list) else [tsk.neumann_linear]
        u_dofs = np.zeros((tsk.basis.N, tsk.n_tasks))
        filter_radius = cfg.filter_radius.init_value \
            if isinstance(
                cfg.filter_radius.num_steps, (int, float)
            ) else cfg.filter_radius.target_value
        # filter_radius = cfg.filter_radius.init \
        #     if cfg.filter_radius_step > 0 else cfg.filter_radius
        return (
            iter_begin,
            iter_end,
            rho,
            rho_prev,
            rho_filtered,
            rho_projected,
            dH_drho,
            grad_filtered,
            dC_drho_projected,
            energy_mean,
            dC_drho_full,
            dC_drho_design_eles,
            scaling_rate,
            rho_design_eles,
            rho_clip_lower,
            rho_clip_upper,
            u_dofs,
            filter_radius
        )

    def optimize(self):
        tsk = self.tsk
        cfg = self.cfg
        tsk.export_analysis_condition_on_mesh(cfg.dst_path)
        logger.info(f"dst_path : {cfg.dst_path}")
        self.init_schedulers()
        density_interpolation, dC_drho_func = interpolation_funcs(cfg)
        (
            iter_begin,
            iter_end,
            rho,
            rho_prev,
            rho_filtered,
            rho_projected,
            dH_drho,
            grad_filtered,
            dC_drho_projected,
            energy_mean,
            dC_drho_full,
            dC_drho_design_eles,
            scaling_rate,
            rho_design_eles,
            rho_clip_lower,
            rho_clip_upper,
            u_dofs,
            filter_radius
        ) = self.initialize_params()
        elements_volume_design = tsk.elements_volume[tsk.design_elements]
        elements_volume_design_sum = np.sum(elements_volume_design)
        self.filter.update_radius(filter_radius)

        # Loop 1 - N
        for iter_num in range(iter_begin, iter_end):
            logger.info(f"iterations: {iter_num} / {iter_end - 1}")
            (
                p,
                vol_frac,
                beta,
                move_limit,
                eta, percentile, filter_radius
            ) = self.schedulers.values_as_list(
                iter_num,
                [
                    'p', 'vol_frac', 'beta', 'move_limit',
                    'eta', 'percentile', 'filter_radius'
                ],
                export_log=True,
                precision=6
            )

            if filter_radius != self.filter.radius:
                logger.info("--- Filter Update ---")
                self.filter.update_radius(filter_radius)

            logger.info("--- project and filter ---")
            rho_prev[:] = rho[:]
            rho_filtered[:] = self.filter.forward(rho)
            projection.heaviside_projection_inplace(
                rho_filtered, beta=beta, eta=cfg.beta_eta, out=rho_projected
            )
            logger.info("--- compute compliance ---")
            dC_drho_design_eles[:] = 0.0
            dC_drho_full[:] = 0.0
            energy_mean[:] = 0.0
            u_max = list()

            compliance_avg = self.fem.objectives_multi_load(
                rho_projected, p, u_dofs
            ).mean()
            energy = self.fem.energy_multi_load(
                rho_projected, p, u_dofs,
            )
            energy_mean = energy.mean(axis=1)
            for task_loop in range(self.tsk.n_tasks):

                u_max.append(np.abs(u_dofs[:, task_loop]).max())
                dH_drho[:] = 0.0
                np.copyto(
                    dC_drho_projected,
                    dC_drho_func(
                        rho_projected,
                        energy[:, task_loop],
                        self.tsk.material_coef,
                        self.tsk.material_coef*cfg.E_min_coeff,
                        p
                    )
                )
                projection.heaviside_projection_derivative_inplace(
                    rho_filtered,
                    beta=beta, eta=cfg.beta_eta, out=dH_drho
                )
                np.multiply(dC_drho_projected, dH_drho, out=grad_filtered)
                dC_drho_full[:] += self.filter.gradient(grad_filtered)

            # print(f"dC_drho_full min/max {dC_drho_full.min()} {dC_drho_full.max()}")
            dC_drho_full /= tsk.n_tasks
            if cfg.sensitivity_filter:
                logger.info("--- sensitivity filter ---")
                filtered = self.filter.forward(dC_drho_full)
                np.copyto(dC_drho_full, filtered)

            dC_drho_design_eles[:] = dC_drho_full[tsk.design_elements]
            rho_design_eles[:] = rho[tsk.design_elements]
            logger.info("--- update density ---")
            self.rho_update(
                iter_num,
                rho_design_eles,
                rho_projected,
                dC_drho_design_eles,
                u_dofs,
                energy_mean,
                scaling_rate,
                move_limit,
                eta,
                beta,
                rho_clip_lower,
                rho_clip_upper,
                percentile,
                elements_volume_design,
                elements_volume_design_sum,
                vol_frac
            )
            rho[tsk.design_elements] = rho_design_eles
            if cfg.design_dirichlet is True:
                rho[tsk.neumann_elements] = 1.0
            else:
                rho[tsk.dirichlet_neumann_elements] = 1.0

            self.recorder.feed_data(
                "rho_projected", rho_projected[tsk.design_elements]
            )
            self.recorder.feed_data("energy", energy_mean)
            self.recorder.feed_data("compliance", compliance_avg)
            self.recorder.feed_data("scaling_rate", scaling_rate)
            u_max = u_max[0] if len(u_max) == 1 else np.array(u_max)
            self.recorder.feed_data("u_max", u_max)

            if any(
                (iter_num % (cfg.max_iters // cfg.record_times) == 0,
                 iter_num == 1)
            ):
                logger.info(f"Saving at iteration {iter_num}")
                self.recorder.print()
                self.recorder.export_progress()

                visualization.export_mesh_with_info(
                    tsk.mesh,
                    cell_data_names=["rho_projected", "energy"],
                    cell_data_values=[rho_projected, energy_mean],
                    filepath=cfg.vtu_path(iter_num)
                )
                visualization.write_mesh_with_info_as_image(
                    mesh_path=cfg.vtu_path(iter_num),
                    mesh_scalar_name="rho_projected",
                    clim=(0.0, 1.0),
                    image_path=cfg.image_path(iter_num, "rho_projected"),
                    image_title=f"Iteration : {iter_num}"
                )
                visualization.write_mesh_with_info_as_image(
                    mesh_path=cfg.vtu_path(iter_num),
                    mesh_scalar_name="energy",
                    clim=(0.0, np.max(energy)),
                    image_path=cfg.image_path(iter_num, "energy"),
                    image_title=f"Iteration : {iter_num}"
                )
                np.savez_compressed(
                    f"{cfg.dst_path}/data/{str(iter_num).zfill(6)}-rho.npz",
                    rho_design_elements=rho[tsk.design_elements]
                )

        if cfg.scaling is True:
            self.unscale()
        visualization.rho_histo_plot(
            rho[tsk.design_elements],
            f"{self.cfg.dst_path}/mesh_rho/last.jpg"
        )
        visualization_mesh.export_submesh(
            tsk, rho, 0.5, f"{cfg.dst_path}/cubic_top.vtu"
        )
        self.recorder.export_histories(fname="histories.npz")

    def rho_update(
        self,
        iter_num: int,
        rho_design_eles: np.ndarray,
        rho_projected: np.ndarray,
        dC_drho_design_eles: np.ndarray,
        u_dofs: np.ndarray,
        energy_mean: np.ndarray,
        scaling_rate: np.ndarray,
        move_limit: float,
        eta: float,
        rho_clip_lower: np.ndarray,
        rho_clip_upper: np.ndarray,
        lambda_lower: float,
        lambda_upper: float,
        percentile: float | None,
        elements_volume_design: np.ndarray,
        elements_volume_design_sum: float,
        vol_frac: float
    ):
        raise NotImplementedError("")
