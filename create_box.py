import sys
import numpy as np


def create_tet_box_gmsh(x_len, y_len, z_len, mesh_size, fpath):
    import gmsh
    print("generate mesh")
    gmsh.initialize()
    gmsh.model.add('plate')
    gmsh.model.occ.addBox(0, 0, 0, x_len, y_len, z_len)
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), mesh_size)

    gmsh.model.mesh.setOrder(1)
    gmsh.model.mesh.generate(3)
    gmsh.write(fpath)
    gmsh.finalize()


def create_hex_box_gmsh(x_len, y_len, z_len, mesh_size, fpath):
    import gmsh
    print("generate Hex mesh")
    gmsh.initialize()
    gmsh.model.add("box")

    # Create the box volume
    box = gmsh.model.occ.addBox(0, 0, 0, x_len, y_len, z_len)
    gmsh.model.occ.synchronize()

    dim, tag = 3, box

    # Transfinite meshing setup
    gmsh.model.mesh.setTransfiniteAutomatic()      # 自動構造格子（推奨）
    gmsh.model.mesh.setRecombine(dim, tag)         # Hex化指示
    gmsh.model.mesh.setAlgorithm(dim, tag, 6)      # ✅ 6 = Frontal (Hex対応)

    # Mesh size
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), mesh_size)

    gmsh.model.mesh.setOrder(1)
    gmsh.model.mesh.generate(3)
    gmsh.write(fpath)

    gmsh.finalize()


def create_structured_hex_box(x_len, y_len, z_len, mesh_size, fpath):
    import math
    import gmsh
    gmsh.initialize()
    gmsh.model.add("structured_hex_box")

    # 分割数（各軸）
    nx = max(1, math.ceil(x_len / mesh_size))
    ny = max(1, math.ceil(y_len / mesh_size))
    nz = max(1, math.ceil(z_len / mesh_size))
    print(f"Divisions: nx={nx}, ny={ny}, nz={nz}")

    # 立方体形状
    box = gmsh.model.occ.addBox(0, 0, 0, x_len, y_len, z_len)
    gmsh.model.occ.synchronize()

    # Transfinite設定
    gmsh.model.mesh.setTransfiniteVolume(box)

    # 各面にも設定
    for dim, tag in gmsh.model.getEntities(2):
        gmsh.model.mesh.setTransfiniteSurface(tag)

    # エッジの分割数設定
    for dim, tag in gmsh.model.getEntities(1):
        try:
            xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(dim, tag)
            length = math.sqrt((xmax - xmin)**2 + (ymax - ymin)**2 + (zmax - zmin)**2)
            div = max(2, math.ceil(length / mesh_size))
            gmsh.model.mesh.setTransfiniteCurve(tag, div)
        except Exception as e:
            print(f"Warning: edge {tag} skipped: {e}")

    # Hexメッシュ化
    gmsh.model.mesh.setRecombine(3, box)
    gmsh.option.setNumber("Mesh.ElementOrder", 1)
    gmsh.option.setNumber("Mesh.RecombineAll", 1)
    # gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)


    # 物理グループ（出力用）
    gmsh.model.addPhysicalGroup(3, [box], 1)
    gmsh.model.setPhysicalName(3, 1, "StructuredHexBox")

    # メッシュ生成・保存
    gmsh.model.mesh.generate(3)
    gmsh.write(fpath)

    # if "-nopopup" not in sys.argv:
    #     gmsh.fltk.run()

    gmsh.finalize()



def compute_divisions_from_mesh_size(x_len, y_len, z_len, mesh_size):
    nx = max(1, int(round(x_len / mesh_size)))
    ny = max(1, int(round(y_len / mesh_size)))
    nz = max(1, int(round(z_len / mesh_size)))
    return nx, ny, nz


def create_hex_box_gmsh_n(x_len, y_len, z_len, mesh_size, fpath):
    nx, ny, nz = compute_divisions_from_mesh_size(x_len, y_len, z_len, mesh_size)

    gmsh.initialize()
    try:
        gmsh.model.add("hexbox")
        box = gmsh.model.occ.addBox(0, 0, 0, x_len, y_len, z_len)
        gmsh.model.occ.synchronize()

        lines = gmsh.model.getEntities(1)
        surfaces = gmsh.model.getEntities(2)
        volumes = gmsh.model.getEntities(3)

        for dim, tag in lines:
            boundary = gmsh.model.getBoundary([(dim, tag)], oriented=True)
            if len(boundary) < 2:
                continue

            start = boundary[0][1]
            end = boundary[1][1]
            p1 = np.array(gmsh.model.getValue(0, start, []))
            p2 = np.array(gmsh.model.getValue(0, end, []))
            vec = p2 - p1
            direction = np.abs(vec / np.linalg.norm(vec))

            if np.allclose(direction, [1, 0, 0]):
                gmsh.model.mesh.setTransfiniteCurve(tag, nx + 1)
            elif np.allclose(direction, [0, 1, 0]):
                gmsh.model.mesh.setTransfiniteCurve(tag, ny + 1)
            elif np.allclose(direction, [0, 0, 1]):
                gmsh.model.mesh.setTransfiniteCurve(tag, nz + 1)

        for dim, tag in surfaces:
            gmsh.model.mesh.setTransfiniteSurface(tag)
            gmsh.model.mesh.setRecombine(dim, tag)

        for dim, tag in volumes:
            gmsh.model.mesh.setTransfiniteVolume(tag)
            gmsh.model.mesh.setRecombine(dim, tag)

        gmsh.model.mesh.generate(3)
        gmsh.write(fpath)

    finally:
        gmsh.finalize()



def create_refined_box_gmsh(
    x_len, y_len, z_len,
    threshold_x_lower, threshold_x_upper,
    size_min, size_max,
    fpath
):
    gmsh.initialize()
    gmsh.model.add("refined_box")

    # 1. 立方体のジオメトリ作成
    gmsh.model.occ.addBox(0, 0, 0, x_len, y_len, z_len)
    gmsh.model.occ.synchronize()

    # 2. 左側の範囲（x < threshold_x_lower）を細かく
    field_left = gmsh.model.mesh.field.add("Box")
    gmsh.model.mesh.field.setNumber(field_left, "VIn", size_min)
    gmsh.model.mesh.field.setNumber(field_left, "VOut", size_max)
    gmsh.model.mesh.field.setNumber(field_left, "XMin", 0)
    gmsh.model.mesh.field.setNumber(field_left, "XMax", threshold_x_lower)
    gmsh.model.mesh.field.setNumber(field_left, "YMin", 0)
    gmsh.model.mesh.field.setNumber(field_left, "YMax", y_len)
    gmsh.model.mesh.field.setNumber(field_left, "ZMin", 0)
    gmsh.model.mesh.field.setNumber(field_left, "ZMax", z_len)

    # 3. 右側の範囲（x > threshold_x_upper）を細かく
    field_right = gmsh.model.mesh.field.add("Box")
    gmsh.model.mesh.field.setNumber(field_right, "VIn", size_min)
    gmsh.model.mesh.field.setNumber(field_right, "VOut", size_max)
    gmsh.model.mesh.field.setNumber(field_right, "XMin", threshold_x_upper)
    gmsh.model.mesh.field.setNumber(field_right, "XMax", x_len)
    gmsh.model.mesh.field.setNumber(field_right, "YMin", 0)
    gmsh.model.mesh.field.setNumber(field_right, "YMax", y_len)
    gmsh.model.mesh.field.setNumber(field_right, "ZMin", 0)
    gmsh.model.mesh.field.setNumber(field_right, "ZMax", z_len)

    # 4. 2つのフィールドを統合（最小値を採用）
    field_min = gmsh.model.mesh.field.add("Min")
    gmsh.model.mesh.field.setNumbers(field_min, "FieldsList", [field_left, field_right])
    gmsh.model.mesh.field.setAsBackgroundMesh(field_min)

    # 5. メッシュ生成と保存
    gmsh.model.mesh.generate(3)
    gmsh.write(fpath)
    gmsh.finalize()



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description=''
    )
    parser.add_argument(
        '--mesh_size', '-MS', type=float, default=0.2, help=''
    )
    parser.add_argument(
        '--mesh_path', '-MP', type=str, default="plate.msh", help=''
    )
    parser.add_argument(
        '--box_type', '-BT', type=str, default="normal", help=''
    )
    parser.add_argument(
        '--element_type', '-ET', type=str, default="hex", help=''
    )
    parser.add_argument(
        '--task_name', '-TN', type=str, default="down", help=''
    )
    args = parser.parse_args()
    if args.task_name == "down":
        x_len = 4.0
        y_len = 0.16
        z_len = 2.0
    elif args.task_name == "down_box":
        x_len = 4.0
        y_len = 3.0
        z_len = 2.0
    elif args.task_name == "pull":
        x_len = 8.0
        y_len = 3.0
        z_len = 0.5
    elif args.task_name == "pull_2":
        x_len = 4.0
        y_len = 2.0
        z_len = 0.5
    # x_len = 4.0
    # y_len = 3.0
    # z_len = 2.0
    # z_len = 0.5
    mesh_size = args.mesh_size
    
    if args.element_type == "hex":
        # create_hex_box_gmsh(x_len, y_len, z_len, mesh_size, args.mesh_path)
        
        if args.box_type == "skfem":
            from sktopt.mesh import toy_problem
            mesh = toy_problem.create_box_hex(x_len, y_len, z_len, mesh_size)
            mesh.save(args.mesh_path)
        elif args.box_type == "gmsh":
            create_structured_hex_box(
                x_len, y_len, z_len,
                mesh_size,
                args.mesh_path
            )
    else:
        if args.box_type == "skfem":
            from sktopt.mesh.toy_problem import create_box
            mesh = create_box(x_len, y_len, z_len, mesh_size)
            mesh.save(args.mesh_path)
        elif args.box_type == "gmsh":
            create_tet_box_gmsh(x_len, y_len, z_len, mesh_size, args.mesh_path)
        else:
            create_refined_box_gmsh(
                x_len, y_len, z_len,
                threshold_x_lower=0.05*x_len,
                threshold_x_upper=0.95*x_len,
                size_min=args.mesh_size,
                size_max=args.mesh_size*1.5,
                fpath=args.mesh_path
            )
