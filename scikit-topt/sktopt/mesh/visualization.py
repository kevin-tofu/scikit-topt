import numpy as np
import sktopt


def export_submesh(
    tsk: sktopt.mesh.TaskConfig,
    rho_projected: np.ndarray,
    threshold: float,
    dst_path: str
):
    mesh = tsk.mesh
    remove_elements = tsk.design_elements[
        rho_projected[tsk.design_elements] <= threshold
    ]
    kept_elements = sktopt.mesh.task.setdiff1d(
        tsk.all_elements, remove_elements
    )
    kept_t = mesh.t[:, kept_elements]
    unique_vertex_indices = np.unique(kept_t)
    new_points = np.ascontiguousarray(mesh.p[:, unique_vertex_indices])
    index_map = {
        old: new for new, old in enumerate(unique_vertex_indices)
    }
    new_elements = np.vectorize(index_map.get)(kept_t)
    new_elements = np.ascontiguousarray(new_elements)
    meshtype = type(mesh)
    submesh = meshtype(new_points, new_elements)
    submesh.save(dst_path)

