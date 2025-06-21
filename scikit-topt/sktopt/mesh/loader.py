import pathlib
import numpy as np

import meshio
import skfem
from skfem.models.elasticity import linear_elasticity


def basis_from_file(
    mesh_path: str,
    intorder: int = 2
) -> skfem.Basis:
    _mesh_path = pathlib.Path(mesh_path)
    mesh = meshio.read(_mesh_path)
    cell_types = {cell.type for cell in mesh.cells}

    if "hexahedron" in cell_types:
        mesh_skfem = skfem.MeshHex.load(_mesh_path).oriented()
        element = skfem.ElementVector(skfem.ElementHex1())
    elif "tetra" in cell_types:
        mesh_skfem = skfem.MeshTet.load(_mesh_path).oriented()
        element = skfem.ElementVector(skfem.ElementTetP1())
    else:
        raise ValueError(f"Unsupported cell types: {cell_types}")

    basis = skfem.Basis(
        mesh_skfem, element,
        intorder=intorder
    )
    return basis


def load_stiffness_matrix(
    mesh_path: str,
    E: float = 1.0,
    nu: float = 0.3
) -> np.ndarray:
    basis = basis_from_file(mesh_path)
    stiffness_mat = skfem.asm(linear_elasticity(E, nu), basis)
    return stiffness_mat


if __name__ == '__main__':
    import argparse
    from scipy.sparse import save_npz

    parser = argparse.ArgumentParser(
        description='Assemble stiffness matrix from a mesh file.'
    )
    parser.add_argument(
        'mesh_path', type=str, help='Path to the mesh file (.msh)'
    )
    parser.add_argument(
        '--E', type=float, default=1.0, help='Young\'s modulus'
    )
    parser.add_argument(
        '--nu', type=float, default=0.3, help='Poisson\'s ratio'
    )
    parser.add_argument(
        '--output', type=str, default='stiffness_matrix.npz',
        help='Output filename for the stiffness matrix'
    )

    args = parser.parse_args()
    stiffness_mat = load_stiffness_matrix(args.mesh_path, E=args.E, nu=args.nu)
    save_npz(args.output, stiffness_mat)
    print(f"Stiffness matrix saved to {args.output}")
