import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from dataclasses import dataclass, asdict
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
from sktopt.fea import solver
from sktopt import filters
from sktopt.fea import composer
from sktopt.core import misc
from sktopt.tools.logconf import mylogger
logger = mylogger(__name__)
import logging
logging.getLogger("skfem").setLevel(logging.WARNING)


@dataclass
class DensityMethodConfig():
    """
    Configuration for density-based topology optimization (SIMP/RAMP).

    This configuration controls interpolation, continuation schedules via
    `SchedulerConfig`, filtering/projection thresholds, and numerical options.
    It targets sensitivity-based methods such as SIMP and RAMP.

    Notes on continuation
    ---------------------
    Parameters that change during optimization are represented by
    :class:`tools.SchedulerConfig`:
      - ``p``: penalization power (e.g., 1.0 → 3.0)
      - ``vol_frac``: volume fraction constraint (e.g., 0.8 → 0.4)
      - ``beta``: Heaviside projection sharpness (e.g., 1.0 → 2.0)
      - ``filter_radius``: filter radius (often constant)

    Each scheduler defines how a value evolves (e.g., "Step", "StepAccelerating",
    "Constant") and may include extra shape parameters (e.g., ``curvature`` for
    accelerating schedules).

    Attributes
    ----------
    dst_path : str
        Output directory for results and logs.
    interpolation : {"SIMP", "RAMP"}
        Material interpolation model for penalization.
    record_times : int
        Number of snapshots recorded during optimization.
    max_iters : int
        Maximum number of optimization iterations.

    beta_eta : float
        Threshold parameter (η) for Heaviside projection; the density at which
        the projection transitions sharply between 0 and 1.
    eta : float
        Threshold density used in projection or post-processing.

    p : tools.SchedulerConfig
        Continuation schedule for penalization power.
        Default: Step from 1.0 to 3.0.
    vol_frac : tools.SchedulerConfig
        Continuation schedule for volume fraction constraint.
        Default: Step from 0.8 to 0.4.
    beta : tools.SchedulerConfig
        Continuation schedule for Heaviside sharpness (β).
        Default: StepAccelerating from 1.0 to 2.0 with ``curvature=2.0``.
    filter_radius : tools.SchedulerConfig
        Schedule for filter radius. Default: Constant (target_value=1.2).

    E0 : float
        Young's modulus of the solid material.
    E_min : float
        Minimum Young's modulus for void regions.
    rho_min : float
        Minimum density clamp (avoids singular stiffness).
    rho_max : float
        Maximum density clamp (typically 1.0).

    restart : bool
        If True, resume optimization from saved state.
    restart_from : int
        Iteration index to resume from (use -1 to auto-detect latest).

    export_img : bool
        Whether to export images during optimization.
    export_img_opaque : bool
        If True, use opaque rendering for exported images.

    design_dirichlet : bool
        If True, Dirichlet boundary elements are included in the design domain.
    sensitivity_filter : bool
        If True, apply filtering to sensitivity fields.

    solver_option : {"spsolve", "cg_pyamg"}
        Linear solver for the state analysis. ``"cg_pyamg"`` enables multigrid
        acceleration via PyAMG.
    scaling : bool
        If True, apply length/force scaling to normalize geometry and loads for
        improved conditioning.
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
            target_value=0.2
        )
    )
    E0: float = 210e9
    E_min: float = 210e6
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
    Configuration class for controlling parameters in topology optimization.

    This class defines optimization parameters, filtering and projection settings,
    continuation schedules, and numerical solver options. It is designed to support
    sensitivity-based topology optimization algorithms such as SIMP and RAMP.

    Attributes
    ----------

    percentile_init : float
        Initial percentile used to scale sensitivity fields.
    percentile : float
        Final percentile value. If set to a negative value,
        percentile-based scaling is disabled, resulting in a
        more "exact" optimization behavior.
    percentile_step : int
        Number of continuation steps for the percentile value.
        If negative, fixed at percentile_init.
    move_limit_init : float
        Initial move limit for density update.
    move_limit : float
        Final move limit.
    move_limit_step : int
        Number of continuation steps for move limit.
        If negative, move limit is fixed.
    lambda_lower : float
        Lower bound for the Lagrange multiplier in constrained optimization.
    lambda_upper : float
        Upper bound for the Lagrange multiplier.

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
        strain_energy_mean: np.ndarray,
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
    Base class for sensitivity-based topology optimization routines.

    This class provides common functionality shared by multiple optimization \
        strategies
    (e.g., OC, LDMOC, EUMOC), including finite element analysis (FEA), \
        sensitivity
    filtering, and density projection using Heaviside-type functions. It does \
        not
    perform the optimization update itself — that logic is delegated to \
        subclasses.

    Typical usage involves:
    - Assembling the global stiffness matrix
    - Solving the displacement field under given loads and boundary conditions
    - Computing the compliance and its sensitivity with respect to density
    - Applying density or sensitivity filters (e.g., Helmholtz, cone filters)
    - Projecting intermediate densities using smooth Heaviside functions

    This class is designed for reuse and extension. All algorithm-specific \
        update
    steps (e.g., density changes, Lagrange multiplier updates) should be \
        implemented
    in derived classes such as `OC_Optimizer`, `LDMOC_Optimizer`, or `\
        EUMOC_Optimizer`.

    Responsibilities
    ----------------
    - Manage problem configuration (e.g., material parameters, mesh, basis)
    - Provide filtered sensitivities for stable optimization
    - Apply projection functions for sharp interface control
    - Evaluate objective functions such as compliance

    Notes
    -----
    This class serves as the backbone of sensitivity-based topology \
        optimization.
    Subclasses are expected to override methods such as `rho_update()` \
        to implement specific optimization logic.

    Attributes
    ----------
    tsk : TaskConfig
        Contains FEM mesh, basis, boundary condition data, and load vectors.
    config : DensityMethodConfig
        Holds numerical and algorithmic settings like filtering radius,
        penalization power, projection beta, etc.

    """

    def __init__(
        self,
        cfg: DensityMethodConfig,
        tsk: sktopt.mesh.TaskConfig,
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

        # self.recorder = self.add_recorder(tsk)
        self.schedulers = tools.Schedulers(self.cfg.dst_path)

    def add_recorder(
        self, tsk: sktopt.mesh.TaskConfig
    ) -> tools.HistoriesLogger:
        recorder = tools.HistoriesLogger(self.cfg.dst_path)
        recorder.add("rho_projected", plot_type="min-max-mean-std")
        recorder.add("strain_energy", plot_type="min-max-mean-std")
        recorder.add("vol_error")
        if isinstance(tsk.force, list):
            recorder.add("u_max", plot_type="min-max-mean-std")
        else:
            recorder.add("u_max")
        recorder.add("compliance", ylog=True)
        recorder.add("scaling_rate", plot_type="min-max-mean-std")
        return recorder

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
            rho[tsk.force_elements] = 1.0
        else:
            rho[tsk.dirichlet_force_elements] = 1.0
        rho[tsk.fixed_elements] = 1.0
        return rho, iter_begin, iter_end

    def initialize_params(self):
        """
            Initialize parameters and preallocate work arrays for density-based topology 
            optimization using the SIMP method.

            This function prepares all arrays required for iterative optimization. 
            By preallocating memory and reusing arrays in-place, repeated allocations 
            are avoided and execution speed is improved.

            Returns
            -------
            iter_begin : int
                Starting iteration index.
            iter_end : int
                Final iteration index.
            rho : ndarray of shape (n_elements,)
                Current element-wise design density variables.
            rho_prev : ndarray of shape (n_elements,)
                Copy of densities from the previous iteration.
            rho_filtered : ndarray of shape (n_elements,)
                Densities after applying the density filter (e.g., Helmholtz filter).
            rho_projected : ndarray of shape (n_elements,)
                Densities after applying the projection function (e.g., Heaviside projection).
            dH_drho : ndarray of shape (n_elements,)
                Derivative of the projection function with respect to the filtered densities.
            grad_filtered : ndarray of shape (n_elements,)
                Gradient field after combining sensitivity with projection derivative.
            dC_drho_projected : ndarray of shape (n_elements,)
                Compliance sensitivities with respect to the projected densities.
            strain_energy_mean : ndarray of shape (n_elements,)
                Average element strain energy over all load cases.
            dC_drho_full : ndarray of shape (n_elements,)
                Compliance sensitivities mapped back to the full set of elements.
            dC_drho_design_eles : ndarray of shape (n_design_elements,)
                Compliance sensitivities restricted to design elements only.
            scaling_rate : ndarray of shape (n_design_elements,)
                Scaling factors used in the update scheme (e.g., MOC/OC update).
            rho_design_eles : ndarray of shape (n_design_elements,)
                Subset of densities corresponding to design elements only.
            rho_clip_lower : ndarray of shape (n_design_elements,)
                Lower clipping bounds for density updates.
            rho_clip_upper : ndarray of shape (n_design_elements,)
                Upper clipping bounds for density updates.
            force_vec_list : list of ndarray
                List of external force vectors, one for each load case.
            u_dofs : ndarray of shape (ndof, n_load_cases)
                Displacement field solutions for each degree of freedom
                and each load case.
            filter_radius : float
                Initial radius of the density filter. Determined either from 
                a fixed value or from a scheduled update.

            Notes
            -----
            - This setup is specific to density-based SIMP topology optimization.
            - Arrays are reused in-place across iterations to minimize memory 
            allocation overhead.
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
        strain_energy_mean = np.zeros_like(rho)
        dC_drho_full = np.zeros_like(rho)
        dC_drho_design_eles = np.zeros_like(rho[tsk.design_elements])
        scaling_rate = np.empty_like(rho[tsk.design_elements])
        rho_design_eles = np.empty_like(rho[tsk.design_elements])
        rho_clip_lower = np.empty_like(rho[tsk.design_elements])
        rho_clip_upper = np.empty_like(rho[tsk.design_elements])
        force_vec_list = tsk.force \
            if isinstance(tsk.force, list) else [tsk.force]
        u_dofs = np.zeros((tsk.basis.N, len(force_vec_list)))
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
            strain_energy_mean,
            dC_drho_full,
            dC_drho_design_eles,
            scaling_rate,
            rho_design_eles,
            rho_clip_lower,
            rho_clip_upper,
            force_vec_list,
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
            strain_energy_mean,
            dC_drho_full,
            dC_drho_design_eles,
            scaling_rate,
            rho_design_eles,
            rho_clip_lower,
            rho_clip_upper,
            force_vec_list,
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
            strain_energy_mean[:] = 0.0
            u_max = list()

            compliance_avg = solver.compute_compliance_basis_multi_load(
                tsk.basis, tsk.free_dofs, tsk.dirichlet_dofs, force_vec_list,
                cfg.E0, cfg.E_min, p, tsk.nu,
                rho_projected,
                u_dofs,
                elem_func=density_interpolation,
                solver=cfg.solver_option,
                n_joblib=cfg.n_joblib
            ).mean()
            strain_energy = composer.strain_energy_skfem_multi(
                tsk.basis, rho_projected, u_dofs,
                cfg.E0, cfg.E_min, p, tsk.nu,
                elem_func=density_interpolation
            )
            strain_energy_mean = strain_energy.mean(axis=1)
            for force_loop, _ in enumerate(force_vec_list):

                u_max.append(np.abs(u_dofs[:, force_loop]).max())
                dH_drho[:] = 0.0
                np.copyto(
                    dC_drho_projected,
                    dC_drho_func(
                        rho_projected,
                        strain_energy[:, force_loop],
                        cfg.E0, cfg.E_min, p
                    )
                )
                projection.heaviside_projection_derivative_inplace(
                    rho_filtered,
                    beta=beta, eta=cfg.beta_eta, out=dH_drho
                )
                np.multiply(dC_drho_projected, dH_drho, out=grad_filtered)
                dC_drho_full[:] += self.filter.gradient(grad_filtered)

            dC_drho_full /= len(force_vec_list)
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
                strain_energy_mean,
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
                rho[tsk.force_elements] = 1.0
            else:
                rho[tsk.dirichlet_force_elements] = 1.0

            self.recorder.feed_data(
                "rho_projected", rho_projected[tsk.design_elements]
            )
            self.recorder.feed_data("strain_energy", strain_energy_mean)
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
                    cell_data_names=["rho_projected", "strain_energy"],
                    cell_data_values=[rho_projected, strain_energy_mean],
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
                    mesh_scalar_name="strain_energy",
                    clim=(0.0, np.max(strain_energy)),
                    image_path=cfg.image_path(iter_num, "strain_energy"),
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
        strain_energy_mean: np.ndarray,
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
