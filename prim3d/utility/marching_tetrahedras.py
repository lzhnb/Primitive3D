# Copyright (c) Zhihao Liang. All rights reserved.
from typing import List, Optional, Sequence, Tuple, Union

import torch
import numpy as np

import prim3d.libPrim3D as _C


def marching_tetrahedras(
    points: Union[torch.Tensor, np.ndarray],
    tetrahedras: Union[torch.Tensor, np.ndarray],
    sdfs: Union[torch.Tensor, np.ndarray],
    thresh: float,
    verbose: bool = False,
    cpu: bool = False
) -> Tuple[torch.Tensor]:
    """python wrapper of marching tetrahedras
    Args:
        thresh (float): thresh of marching cubes
        verbose (bool, optional):
            print verbose informations or not. Defaults to False.
        cpu (bool, optional):
            cpu mode(wrapper of mcubes). Defaults to False.
    Returns:
        Tuple[torch.Tensor]: vertices and faces
    """
    if cpu or not torch.cuda.is_available():
        raise NotImplementedError
    else:
        if isinstance(points, np.ndarray):
            points = torch.tensor(points)
        if isinstance(tetrahedras, np.ndarray):
            tetrahedras = torch.tensor(tetrahedras)
        if isinstance(sdfs, np.ndarray):
            sdfs = torch.tensor(sdfs)
        points = points.cuda()
        tetrahedras = tetrahedras.cuda()
        sdfs = sdfs.cuda()
        vertices, faces = _C.marching_tetrahedras(points, tetrahedras, sdfs, thresh)

    if verbose:
        print(f"#vertices={vertices.shape[0]}")
        print(f"#triangles={faces.shape[0]}")

    return vertices, faces

