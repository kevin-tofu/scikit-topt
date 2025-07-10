import os
from typing import Literal
import inspect
import shutil
import json
from dataclasses import dataclass, asdict
import numpy as np
import sktopt
from sktopt import tools
from sktopt.core import derivatives, projection
from sktopt.core import visualization
from sktopt.mesh import visualization as visualization_mesh
from sktopt.fea import solver
from sktopt import filter
from sktopt.fea import composer
from sktopt.core import misc
from sktopt.tools.logconf import mylogger
logger = mylogger(__name__)



@dataclass
class DensityMethodConfig():
    """
    Configuration class for controlling parameters in topology optimization.

    This class defines optimization parameters, filtering and projection settings,
    continuation schedules, and numerical solver options. It is designed to support
    sensitivity-based topology optimization algorithms such as SIMP and RAMP.

    Attributes
    ----------
    dst_path : str
        Directory to which results and logs are saved.
    interpolation : {"SIMP", "RAMP"}
        Interpolation method for material penalization.
    record_times : int
        Number of snapshots to record during optimization.
    max_iters : int
        Maximum number of optimization iterations.

    p_init : float
        Initial penalization power in SIMP/RAMP interpolation.
    p : float
        Final value of the penalization power.
    p_step : int
        Number of steps to continue p from p_init to p.
        If negative, p is fixed at p_init.

    vol_frac_init : float
        Initial volume fraction for continuation.
    vol_frac : float
        Final volume fraction constraint.
    vol_frac_step : int
        Number of continuation steps from vol_frac_init to vol_frac.
        If negative, vol_frac is fixed at vol_frac_init.

    beta_init : float
        Initial sharpness parameter for Heaviside projection.
    beta : float
        Final beta value for projection.
    beta_step : int
        Number of continuation steps from beta_init to beta.
        If negative, beta is fixed at beta_init.

    beta_curvature : float
        Controls the curvature of the Heaviside projection function.
    beta_eta : float
        Threshold parameter for Heaviside projection; determines where
        the projection function switches rapidly between 0 and 1.

    eta : float
        Threshold density used in projection. Often used to define
        intermediate region in Heaviside or filtering.

    percentile_init : float
        Initial percentile used to scale sensitivity fields.
    percentile : float
        Final percentile value. If set to a negative value,
        percentile-based scaling is disabled, resulting in a
        more "exact" optimization behavior.
    percentile_step : int
        Number of continuation steps for the percentile value.
        If negative, fixed at percentile_init.

    filter_radius_init : float
        Initial radius of the sensitivity or density filter.
    filter_radius : float
        Final filter radius.
    filter_radius_step : int
        Number of continuation steps for filter radius.
        If negative, filter_radius is fixed at initial value.

    E0 : float
        Young's modulus of the solid material.
    E_min : float
        Minimum Young's modulus (used for void regions).
    rho_min : float
        Minimum density value (to avoid singular stiffness).
    rho_max : float
        Maximum density value (usually 1.0 for full material).

    restart : bool
        Whether to resume optimization from an existing state.
    restart_from : int
        Iteration index to restart from.

    export_img : bool
        Whether to export images during optimization.
    export_img_opaque : bool
        If True, image export uses opaque rendering.

    design_dirichlet : bool
        If True, Dirichlet boundary elements are included in the design domain.

    sensitivity_filter : bool
        If True, applies filtering to the sensitivity field.

    solver_option : {"spsolve", "cg_pyamg"}
        Linear solver to be used in analysis. `"cg_pyamg"` enables multigrid
        acceleration.

    scaling : bool
        If True, length and force scaling are applied to normalize the mesh geometry and load magnitudes.
        This helps improve numerical stability by making the system dimensionless or better conditioned.

    """

    dst_path: str = "./result/pytests"
    interpolation: Literal["SIMP", "RAMP"] = "SIMP"
    record_times: int = 20
    max_iters: int = 200
    p_init: float = 1.0
    p: float = 3.0
    p_step: int = -1
    vol_frac_init: float = 0.8
    vol_frac: float = 0.4
    vol_frac_step: int = -3
    beta_init: float = 1.0
    beta: float = 2
    beta_step: int = -1
    beta_curvature: float = 2.0
    beta_eta: float = 0.50
    eta: float = 0.6
    filter_radius_init: float = 2.0
    filter_radius: float = 1.20
    filter_radius_step: int = -3
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

    def vtu_path(self, iter: int):
        return f"{self.dst_path}/mesh_rho/info_mesh-{iter:08d}.vtu"

    def image_path(self, iter: int, prefix: str):
        if self.export_img:
            return f"{self.dst_path}/mesh_rho/info_{prefix}-{iter:08d}.jpg"
        else:
            return None


@dataclass
class DensityMethodLagrangianConfig(DensityMethodConfig):
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
    percentile_init: float = 60
    percentile: float = -90
    percentile_step: int = -1
    move_limit_init: float = 0.3
    move_limit: float = 0.10
    move_limit_step: int = -3
    lambda_lower: float = 1e-7
    lambda_upper: float = 1e+7


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


from abc import ABC, abstractmethod

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
        iter_loop: int,
        rho_candidate: np.ndarray,
        rho_projected: np.ndarray,
        dC_drho_ave: np.ndarray,
        u_dofs: np.ndarray,
        strain_energy_ave: np.ndarray,
        scaling_rate: np.ndarray,
        move_limit: float,
        eta: float,
        tmp_lower: np.ndarray,
        tmp_upper: np.ndarray,
        lambda_lower: float,
        lambda_upper: float,
        percentile: float,
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
        p_init = cfg.p_init
        vol_frac_init = cfg.vol_frac_init
        move_limit_init = cfg.move_limit_init
        beta_init = cfg.beta_init
        self.schedulers.add(
            "p",
            p_init,
            cfg.p,
            cfg.p_step,
            cfg.max_iters
        )
        self.schedulers.add(
            "vol_frac",
            vol_frac_init,
            cfg.vol_frac,
            cfg.vol_frac_step,
            cfg.max_iters
        )
        self.schedulers.add_object(
            tools.SchedulerSawtoothDecay(
                "move_limit",
                move_limit_init,
                cfg.move_limit,
                cfg.move_limit_step,
                cfg.max_iters
            )
        )
        self.schedulers.add_object(
            tools.SchedulerStepAccelerating(
                "beta",
                beta_init,
                cfg.beta,
                cfg.beta_step,
                cfg.max_iters,
                cfg.beta_curvature,
            )
        )
        self.schedulers.add(
            "percentile",
            cfg.percentile_init,
            cfg.percentile,
            cfg.percentile_step,
            cfg.max_iters
        )
        self.schedulers.add(
            "filter_radius",
            cfg.filter_radius_init,
            cfg.filter_radius,
            cfg.filter_radius_step,
            cfg.max_iters
        )
        if "eta_init" in cfg.__dataclass_fields__:
            self.schedulers.add(
                "eta",
                self.cfg.eta_init,
                self.cfg.eta,
                self.cfg.eta_step,
                self.cfg.max_iters
            )
        else:
            self.schedulers.add(
                "eta",
                self.cfg.eta,
                self.cfg.eta,
                -1,
                self.cfg.max_iters
            )
        if export:
            self.schedulers.export()

    def parameterize(self):
        self.helmholz_solver = filter.HelmholtzFilter.from_defaults(
            self.tsk.mesh,
            self.cfg.filter_radius,
            solver_option="cg_pyamg",
            # solver_option=self.cfg.solver_option,
            dst_path=f"{self.cfg.dst_path}/data",
        )

    def load_parameters(self):
        self.helmholz_solver = filter.HelmholtzFilter.from_file(
            f"{self.cfg.dst_path}/data"
        )

    def initialize_density(self):
        tsk = self.tsk
        cfg = self.cfg
        val_init = cfg.vol_frac_init
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
                iter, data_path = misc.find_latest_iter_file(
                    f"{cfg.dst_path}/data"
                )
                data = np.load(data_path)
                iter_begin = iter + 1
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
        tsk = self.tsk
        cfg = self.cfg
        rho, iter_begin, iter_end = self.initialize_density()
        rho_prev = np.zeros_like(rho)
        rho_filtered = np.zeros_like(rho)
        rho_projected = np.zeros_like(rho)
        dH = np.empty_like(rho)
        grad_filtered = np.empty_like(rho)
        dC_drho_projected = np.empty_like(rho)
        strain_energy_ave = np.zeros_like(rho)
        dC_drho_full = np.zeros_like(rho)
        dC_drho_ave = np.zeros_like(rho[tsk.design_elements])
        scaling_rate = np.empty_like(rho[tsk.design_elements])
        rho_candidate = np.empty_like(rho[tsk.design_elements])
        tmp_lower = np.empty_like(rho[tsk.design_elements])
        tmp_upper = np.empty_like(rho[tsk.design_elements])
        force_list = tsk.force if isinstance(tsk.force, list) else [tsk.force]
        u_dofs = np.zeros((tsk.basis.N, len(force_list)))
        filter_radius = cfg.filter_radius_init \
            if cfg.filter_radius_step > 0 else cfg.filter_radius
        return (
            iter_begin,
            iter_end,
            rho,
            rho_prev,
            rho_filtered,
            rho_projected,
            dH,
            grad_filtered,
            dC_drho_projected,
            strain_energy_ave,
            dC_drho_full,
            dC_drho_ave,
            scaling_rate,
            rho_candidate,
            tmp_lower,
            tmp_upper,
            force_list,
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
            dH,
            grad_filtered,
            dC_drho_projected,
            strain_energy_ave,
            dC_drho_full,
            dC_drho_ave,
            scaling_rate,
            rho_candidate,
            tmp_lower,
            tmp_upper,
            force_list,
            u_dofs,
            filter_radius
        ) = self.initialize_params()
        elements_volume_design = tsk.elements_volume[tsk.design_elements]
        elements_volume_design_sum = np.sum(elements_volume_design)
        self.helmholz_solver.update_radius(
            tsk.mesh, filter_radius, solver_option="cg_pyamg"
        )

        # Loop 1 - N
        for iter_loop, iter in enumerate(range(iter_begin, iter_end)):
            logger.info(f"iterations: {iter} / {iter_end - 1}")
            (
                p,
                vol_frac,
                beta,
                move_limit,
                eta, percentile, filter_radius
            ) = self.schedulers.values_as_list(
                iter,
                [
                    'p', 'vol_frac', 'beta', 'move_limit',
                    'eta', 'percentile', 'filter_radius'
                ],
                export_log=True,
                precision=6
            )

            if filter_radius != self.helmholz_solver.radius:
            # if filter_radius_prev != filter_radius:
                logger.info("!!! Filter Update")
                self.helmholz_solver.update_radius(
                    tsk.mesh, filter_radius, "cg_pyamg"
                )

            logger.info("!!! project and filter")
            rho_prev[:] = rho[:]
            rho_filtered[:] = self.helmholz_solver.filter(rho)
            projection.heaviside_projection_inplace(
                rho_filtered, beta=beta, eta=cfg.beta_eta, out=rho_projected
            )
            logger.info("!!! compute compliance")
            dC_drho_ave[:] = 0.0
            dC_drho_full[:] = 0.0
            strain_energy_ave[:] = 0.0
            u_max = list()

            compliance_avg = solver.compute_compliance_basis_multi_load(
                tsk.basis, tsk.free_dofs, tsk.dirichlet_dofs, force_list,
                cfg.E0, cfg.E_min, p, tsk.nu,
                rho_projected,
                u_dofs,
                elem_func=density_interpolation,
                solver=cfg.solver_option,
                n_joblib=cfg.n_joblib
            )
            strain_energy = composer.strain_energy_skfem_multi(
                tsk.basis, rho_projected, u_dofs,
                cfg.E0, cfg.E_min, p, tsk.nu,
                elem_func=density_interpolation
            )
            for force_loop, force in enumerate(force_list):
                dH[:] = 0.0
                u_max.append(np.abs(u_dofs[:, force_loop]).max())
                strain_energy_ave += strain_energy[:, force_loop]

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
                    beta=beta, eta=cfg.beta_eta, out=dH
                )
                np.multiply(dC_drho_projected, dH, out=grad_filtered)
                dC_drho_full[:] += self.helmholz_solver.gradient(grad_filtered)

            dC_drho_full /= len(force_list)
            strain_energy_ave /= len(force_list)
            compliance_avg /= len(force_list)
            if cfg.sensitivity_filter:
                logger.info("!!! sensitivity filter")
                filtered = self.helmholz_solver.filter(dC_drho_full)
                np.copyto(dC_drho_full, filtered)
            dC_drho_ave[:] = dC_drho_full[tsk.design_elements]
            rho_candidate[:] = rho[tsk.design_elements]
            logger.info("!!! update density")
            self.rho_update(
                # iter_loop,
                iter,
                rho_candidate,
                rho_projected,
                dC_drho_ave,
                u_dofs,
                strain_energy_ave,
                scaling_rate,
                move_limit,
                eta,
                beta,
                tmp_lower,
                tmp_upper,
                percentile,
                elements_volume_design,
                elements_volume_design_sum,
                vol_frac
            )
            rho[tsk.design_elements] = rho_candidate
            if cfg.design_dirichlet is True:
                rho[tsk.force_elements] = 1.0
            else:
                rho[tsk.dirichlet_force_elements] = 1.0

            self.recorder.feed_data(
                "rho_projected", rho_projected[tsk.design_elements]
            )
            self.recorder.feed_data("strain_energy", strain_energy_ave)
            self.recorder.feed_data("compliance", compliance_avg)
            self.recorder.feed_data("scaling_rate", scaling_rate)
            u_max = u_max[0] if len(u_max) == 1 else np.array(u_max)
            self.recorder.feed_data("u_max", u_max)

            if any(
                (iter % (cfg.max_iters // cfg.record_times) == 0,
                 iter == 1)
            ):
                logger.info(f"Saving at iteration {iter}")
                self.recorder.print()
                self.recorder.export_progress()

                visualization.export_mesh_with_info(
                    tsk.mesh,
                    cell_data_names=["rho_projected", "strain_energy"],
                    cell_data_values=[rho_projected, strain_energy_ave],
                    filepath=cfg.vtu_path(iter)
                )
                visualization.write_mesh_with_info_as_image(
                    mesh_path=cfg.vtu_path(iter),
                    mesh_scalar_name="rho_projected",
                    clim=(0.0, 1.0),
                    image_path=cfg.image_path(iter, "rho_projected"),
                    image_title=f"Iteration : {iter}"
                )
                visualization.write_mesh_with_info_as_image(
                    mesh_path=cfg.vtu_path(iter),
                    mesh_scalar_name="strain_energy",
                    clim=(0.0, np.max(strain_energy)),
                    image_path=cfg.image_path(iter, "strain_energy"),
                    image_title=f"Iteration : {iter}"
                )
                np.savez_compressed(
                    f"{cfg.dst_path}/data/{str(iter).zfill(6)}-rho.npz",
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
        iter_loop: int,
        rho_candidate: np.ndarray,
        rho_projected: np.ndarray,
        dC_drho_ave: np.ndarray,
        u_dofs: np.ndarray,
        strain_energy_ave: np.ndarray,
        scaling_rate: np.ndarray,
        move_limit: float,
        eta: float,
        tmp_lower: np.ndarray,
        tmp_upper: np.ndarray,
        lambda_lower: float,
        lambda_upper: float,
        percentile: float,
        elements_volume_design: np.ndarray,
        elements_volume_design_sum: float,
        vol_frac: float
    ):
        raise NotImplementedError("")
