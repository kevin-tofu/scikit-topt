# Scikit-Topt
## SCikit Topology Optimization

### Install Package
```bash
pip install scitopt
poetry add scitopt
```

### Optimize Toy Problem with command line.
```bash
python ./scitopt/core/optimizer/oc.py \
 --dst_path ./result/test1_oc \
 --p 3.0 \
 --p_rate 12.0 \
 --filter_radius 0.7 \
 --move_limit 0.2 \
 --move_limit_rate 10.0 \
 --vol_frac 0.4 \
 --vol_frac_rate 5.0 \
 --beta 5.0 \
 --beta_rate 1.0 \
 --eta 1.0 \
 --record_times 80 \
 --max_iters 200
```


## Acknowledgements
 My software and research does not exist in a vacuum.
Scikit-Topt is standing on the shoulders of proverbial giants. In particular, I want to thank the following projects for constituting the technical backbone of the project:
 - Scipy
 - Scikit-fem
 - Numba
 - MeshIO
 - Gmsh
 - Topology Optimization Community

