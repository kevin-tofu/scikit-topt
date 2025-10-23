from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Literal

import numpy as np

from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix, csr_matrix, diags

import skfem


def get_element_centers(mesh: skfem.Mesh) -> np.ndarray:
    coords = mesh.p[:, mesh.t]
    centers = np.mean(coords, axis=1)
    return centers


def make_kernel(
    kind: Literal["linear", "quadratic", "gaussian"],
    r_min: float, 
    sigma: Optional[float] = None
) -> Tuple[Callable[[np.ndarray], np.ndarray], float]:
    if kind == "linear":
        def kernel(d):
            return np.maximum(0.0, r_min - d)
        support = r_min

    elif kind == "quadratic":
        def kernel(d):
            x = d / r_min
            w = 1.0 - x * x
            w[d >= r_min] = 0.0
            return w
        support = r_min

    elif kind == "gaussian":
        s = (r_min / 3.0) if sigma is None else float(sigma)
        cutoff = 3.0 * s

        def kernel(d):
            return np.exp(-0.5 * (d / s) ** 2)
        support = cutoff

    else:
        raise ValueError("unknown kernel kind")

    return kernel, support


def build_filter_matrix(
    element_centers: np.ndarray,
    kernel: Callable[[np.ndarray], np.ndarray],
    support_radius: float,
    elem_volume: Optional[np.ndarray] = None,
    volume_correction: bool = True,
    design_mask: Optional[np.ndarray] = None,
) -> csr_matrix:
    c = element_centers
    n_all = c.shape[1]
    if design_mask is None:
        design_mask = np.ones(n_all, dtype=bool)

    design_ids = np.nonzero(design_mask)[0]
    tree = cKDTree(c.T)

    rows, cols, data = [], [], []

    for row_idx, i in enumerate(design_ids):
        js_all = np.asarray(tree.query_ball_point(c[:, i], support_radius), dtype=int)
        js = js_all[design_mask[js_all]]
        if js.size == 0:
            continue

        dij = np.linalg.norm(c[:, js] - c[:, [i]], axis=0)
        w = kernel(dij)

        if volume_correction and elem_volume is not None:
            w = w * elem_volume[js]

        col_idx = np.searchsorted(design_ids, js)

        rows.append(np.full(js.size, row_idx, dtype=int))
        cols.append(col_idx)
        data.append(w)

    if not data:
        raise RuntimeError(
            "No neighbor relations found; check support_radius or design_mask."
        )

    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    data = np.concatenate(data)

    n_design = len(design_ids)
    W = coo_matrix((data, (rows, cols)), shape=(n_design, n_design)).tocsr()

    row_sum = np.array(W.sum(axis=1)).ravel()
    row_sum[row_sum == 0.0] = 1.0
    Dinv = diags(1.0 / row_sum)
    W = Dinv @ W
    return W


@dataclass
class SpacialFilter():
    mesh: skfem.Mesh
    radius: float
    element_centers: np.ndarray
    design_mask: Optional[np.ndarray] = None
    dst_path: Optional[str] = None

    def update_radius(
        self,
        radius: float,
        **args
    ):
        self.radius = radius


    @classmethod
    def from_defaults(
        cls,
        mesh: skfem.Mesh,
        radius: float = 1.0,
        dst_path: Optional[str] = None,
        design_mask: Optional[np.ndarray] = None,
    ) -> 'SpacialFilter':

        element_centers = get_element_centers(mesh)
        return cls(
            mesh=mesh,
            radius=radius,
            element_centers=element_centers,
            design_mask=design_mask,
            dst_path=dst_path,
        )

    @classmethod
    def from_file(cls, dst_path: str):
        raise NotImplementedError("")

    def run(
        self,
        rho_element: np.ndarray
    ) -> np.ndarray:
        kernel, support = make_kernel(
            kind="gaussian", r_min=self.radius
        )
        volume_array = None
        W = build_filter_matrix(
            self.element_centers,
            kernel, support,
            elem_volume=volume_array,
            volume_correction=False,
            design_mask=self.design_mask
        )
        if self.design_mask is None:
            rho_filtered = W @ rho
        else:
            rho_filtered = np.copy(rho_element)
            rho_filtered[self.design_mask] = W @ rho[self.design_mask]
        return rho_filtered


if __name__ == '__main__':

    # import skfem
    import sktopt
    x_len, y_len, z_len = 1.0, 1.0, 1.0
    element_size = 0.1
    e = skfem.ElementVector(skfem.ElementHex1())
    mesh = sktopt.mesh.toy_problem.create_box_hex(
        x_len, y_len, z_len, element_size
    )
    filter = SpacialFilter.from_defaults(
        mesh, radius=0.2
    )
    rho = np.random.rand(mesh.t.shape[1])
    for loop in range(1, 201):
        rho = filter.run(rho)
        rho_var = np.var(rho)
        print(f"loop: {loop} rho_var: {rho_var:04f}")
