
### How to define a task
 This section illustrates a typical workflow for running a topology-optimization problem with scikit-fem as the FEM backend and scikit-topt for problem setup and optimization.
 See more details in Python scripts in this directory.

### Load Mesh file from file.
```Python
import skfem
import sktopt

mesh_path = "./data/model.msh"
basis = sktopt.mesh.loader.basis_from_file(
  mesh_path, intorder=2
)


from skfem.io.json import from_file
mesh = from_file("./data/mesh.json")
```

#### Or Create Mesh
```Python
x_len, y_len, z_len = 1.0, 1.0, 1.0
element_size = 0.1

# import pathlib
# msh_path = "model.msh"
# mesh = skfem.MeshHex.load(pathlib.Path(msh_path))

# define basis
mesh = sktopt.mesh.toy_problem.create_box_hex(
  x_len, y_len, z_len, element_size
)
e = skfem.ElementVector(skfem.ElementHex1())
basis = skfem.Basis(mesh, e, intorder=2)
```

#### Set BCs and Force, and define task
```Python

# Specify Dirichlet Boundary Conditions
dirichlet_in_range = sktopt.mesh.utils.get_points_in_range(
    (0.0, 0.03), (0.0, y_len), (0.0, z_len)
)
force_in_range = sktopt.mesh.utils.get_points_in_range(
  (x_len, x_len),
  (y_len*2/5, y_len*3/5),
  (z_len*2/5, z_len*3/5)
)
boundaries = {
    "dirichlet": dirichlet_in_range,
    "force": force_in_range
}
mesh = mesh.with_boundaries(boundaries)
subdomains = {"design": np.array(range(mesh.nelements))}

mesh = mesh.with_subdomains(subdomains)
e = skfem.ElementVector(skfem.ElementHex1())
basis = skfem.Basis(mesh, e, intorder=2)
dirichlet_dir = "all"
force_dir_type = "u^1"
force_value = 100
# Define it as a task
tsk = sktopt.mesh.task.LinearElastisicity.from_facets(
    210e9,
    0.30,
    basis,
    dirichlet_dir,
    force_dir_type,
    force_value
)
```


### Optimize Toy Problem with Python Script

```Python
import sktopt

tsk = sktopt.mesh.toy_problem.toy1()
cfg = sktopt.core.optimizers.OC_Config(
  p=sktopt.tools.SchedulerConfig.step(
    init_value=1.0, target_value=3.0,
    num_steps=3
  ),
  vol_frac=sktopt.tools.SchedulerConfig.constant(
    target_value=0.6
  ),
)

optimizer = sktopt.core.OC_Optimizer(cfg, tsk)

optimizer.parameterize()
optimizer.optimize()
```


### Optimize Toy Problem with command line.
```bash
OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3  MKL_NUM_THREADS=3 PYTHONPATH=scikit-topt python ./scikit-topt/sktopt/core/optimizers/logmoc.py --dst_path ./result/toy2_moc 
 --interpolation SIMP  --vol_frac 0.40  --eta 0.8  --record_times 100  --max_iters 300  --mu_p 5  --task_name toy2  --solver_option spsolve  --rho_min 1e-2  --E0 210e9  --E_min 210e5  --design_dirichle
t true
```
