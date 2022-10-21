# Copyright (c) Zhihao Liang. All rights reserved.
import os
import torch
import mcubes
import numpy as np

import prim3d

DENSITY_GRID = np.load(os.path.join(os.path.dirname(__file__), "data", "bunny.npy"))
print(f"DENSITY_GRID shape: ({DENSITY_GRID.shape[0]}, {DENSITY_GRID.shape[1]}, {DENSITY_GRID.shape[2]})")

if __name__ == "__main__":
    density_grid_cu = torch.tensor(DENSITY_GRID).cuda()
    with prim3d.Timer("cuda marching cubes: {:.6f}s"):
        vertices_cu, faces_cu = prim3d.marching_cubes(density_grid_cu, 0, verbose=True) # verbose to print the number of vertices and faces
    with prim3d.Timer("prim3d save mesh: {:.6f}s\n"):
        prim3d.save_mesh(vertices_cu, faces_cu, filename="bunny.ply")

    with prim3d.Timer("cpu-mode prim3d marching cubes: {:.6f}s\n"):
        vertices_cpu, faces_cpu = prim3d.marching_cubes(density_grid_cu, 0, cpu=True)

    with prim3d.Timer("cpu marching cubes: {:.6f}s"):
        vertices_c, faces_c = mcubes.marching_cubes(DENSITY_GRID, 0)
    with prim3d.Timer("mcubes save mesh: {:.6f}s"):
        mcubes.export_obj(vertices_c, faces_c, filename="bunny.obj")

    assert((vertices_cu.shape[0] == vertices_c.shape[0]))
    assert((faces_cu.shape[0] == faces_c.shape[0]))
    assert((vertices_cpu.numpy() == vertices_c).all())
    assert((faces_cpu.numpy() == faces_c).all())

