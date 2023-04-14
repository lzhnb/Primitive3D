# Copyright (c) Zhihao Liang. All rights reserved.
import os

import numpy as np
import torch

import prim3d

if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(__file__), "data", "tetrahedra")
    points = torch.from_numpy(np.load(os.path.join(data_dir, "points.npy")))
    sdfs = torch.from_numpy(np.load(os.path.join(data_dir, "sdfs.npy")))
    tets = torch.from_numpy(np.load(os.path.join(data_dir, "tetrahedras.npy"))).long()

    with prim3d.Timer("cpu:"):
        verts, faces = prim3d.marching_tetrahedras(points, tets, sdfs)
    
    points = points.cuda()
    sdfs = sdfs.cuda()
    tets = tets.cuda()
    with prim3d.Timer("gpu:"):
        verts, faces = prim3d.marching_tetrahedras(points, tets, sdfs)
    prim3d.save_mesh(verts, faces, filename="sphere_tetrahedra.ply")

