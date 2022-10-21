# Copyright (c) Zhihao Liang. All rights reserved.
from typing import Any

import torch

import prim3d.libPrim3D as _C

def create_raycaster(
    vertices: torch.Tensor,
    faces: torch.Tensor,
) -> _C.RayCaster:
    """create the ray caster to conduct ray casting

    Args:
        vertices (torch.Tensor, [num_vertices, 3]): vertices tensor
        faces (torch.Tensor, [num_triangles, 3]): faces tensor

    Returns:
        _C.RayCaster: ray caster
    """
    enable_optix = _C.enable_optix
    if enable_optix:
        vertices = vertices.cuda() if not vertices.is_cuda else vertices
        faces = faces.cuda() if not faces.is_cuda else faces
    else:
        vertices = vertices.cpu() if vertices.is_cuda else vertices
        faces = faces.cpu() if faces.is_cuda else faces
    
    return _C.create_raycaster(vertices, faces)
