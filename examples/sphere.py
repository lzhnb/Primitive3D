# Copyright (c) Zhihao Liang. All rights reserved.
import mcubes
import numpy as np
import torch

import prim3d

X, Y, Z = np.mgrid[:200, :200, :200]
DENSITY_GRID = (X - 50)**2 + (Y - 50)**2 + (Z - 50)**2 - 25**2


if __name__ == "__main__":
    density_grid_cu = torch.tensor(DENSITY_GRID).cuda()
    with prim3d.Timer("cuda marching cubes: {:.6f}s"):
        vertices_cu, faces_cu = prim3d.marching_cubes(density_grid_cu, 0, verbose=True) # verbose to print the number of vertices and faces
    with prim3d.Timer("prim3d save mesh: {:.6f}s\n"):
        prim3d.save_mesh(vertices_cu, faces_cu, filename="sphere.ply")

    with prim3d.Timer("cpu-mode prim3d marching cubes: {:.6f}s\n"):
        vertices_cpu, faces_cpu = prim3d.marching_cubes(density_grid_cu, 0, cpu=True)

    with prim3d.Timer("cpu marching cubes: {:.6f}s"):
        vertices_c, faces_c = mcubes.marching_cubes(DENSITY_GRID, 0)
    with prim3d.Timer("mcubes save mesh: {:.6f}s"):
        mcubes.export_obj(vertices_c, faces_c, filename="sphere.obj")

    assert((vertices_cu.shape[0] == vertices_c.shape[0]))
    assert((faces_cu.shape[0] == faces_c.shape[0]))
    assert((vertices_cpu.numpy() == vertices_c).all())
    assert((faces_cpu.numpy() == faces_c).all())

