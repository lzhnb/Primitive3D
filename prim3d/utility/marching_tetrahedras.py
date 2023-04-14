import os
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
import trimesh
import prim3d

# fmt: off
triangle_table = torch.tensor(
    [                             # [0, 1, 2, 3]
        [-1, -1, -1, -1, -1, -1], # [-, -, -, -]
        [ 1,  0,  2, -1, -1, -1], # [+, -, -, -]
        [ 4,  0,  3, -1, -1, -1], # [-, +, -, -]
        [ 1,  4,  2,  1,  3,  4], # [+, +, -, -]
        [ 3,  1,  5, -1, -1, -1], # [-, -, +, -]
        [ 2,  3,  0,  2,  5,  3], # [+, -, +, -]
        [ 1,  4,  0,  1,  5,  4], # [-, +, +, -]
        [ 4,  2,  5, -1, -1, -1], # [+, +, +, -]
        [ 4,  5,  2, -1, -1, -1], # [-, -, -, +]
        [ 4,  1,  0,  4,  5,  1], # [+, -, -, +]
        [ 3,  2,  0,  3,  5,  2], # [-, +, -, +]
        [ 1,  3,  5, -1, -1, -1], # [+, +, -, +]
        [ 4,  1,  2,  4,  3,  1], # [-, -, +, +]
        [ 3,  0,  4, -1, -1, -1], # [+, -, +, +]
        [ 2,  0,  1, -1, -1, -1], # [-, +, +, +]
        [-1, -1, -1, -1, -1, -1], # [+, +, +, +]
    ],
    dtype=torch.long,
)

num_triangles_table = torch.tensor(
    [0, 1, 1, 2, 1, 2, 2, 1, 1, 2, 2, 1, 2, 1, 1, 0],
    dtype=torch.long,
)
base_tet_edges = torch.tensor(
    [
        0, 1,
        0, 2,
        0, 3,
        1, 2,
        1, 3,
        2, 3
    ],
    dtype=torch.long,
)
# fmt: on

v_id = torch.pow(2, torch.arange(4, dtype=torch.long))


@torch.no_grad()
def _orientation_test(vertices: torch.Tensor, tets: torch.Tensor) -> torch.Tensor:
    """orientation test to find the tetrahedras needed to flip

    Args:
        vertices (torch.Tensor): input vertices
        tets (torch.Tensor): input tetrahedras

    Returns:
        torch.Tensor: the mask pointed to flip the tetrahedras
    """
    # correction tetrahedras
    tetrahedras_det = vertices[tets, :]
    tetrahedras_det = F.pad(tetrahedras_det, [1, 0], "constant", 1)
    dets = torch.det(tetrahedras_det)
    flip_mask = dets < 0
    return flip_mask


@torch.no_grad()
def _sort_edges(edges: torch.Tensor) -> torch.Tensor:
    """sort last dimension of edges of shape (E, 2)

    Args:
        edges (torch.Tensor): input edges

    Returns:
        torch.Tensor: sorted edges
    """
    order = (edges[:, 0] > edges[:, 1]).long()
    order = order.unsqueeze(dim=1)

    a = torch.gather(input=edges, index=order, dim=1)
    b = torch.gather(input=edges, index=1 - order, dim=1)

    return torch.cat([a, b], -1)


# modify from kaolin: https://github.com/NVIDIAGameWorks/kaolin/blob/master/kaolin/ops/conversions/tetmesh.py#L113
# add orientation test and correction
def marching_tetrahedras(
    vertices: torch.Tensor,
    tets: torch.Tensor,
    sdf: torch.Tensor,
    return_tet_idx: bool = False,
) -> Tuple[torch.Tensor]:
    r"""Convert discrete signed distance fields encoded on tetrahedral grids to triangle
    meshes using marching tetrahedra algorithm as described in `An efficient method of
    triangulating equi-valued surfaces by using tetrahedral cells`_. The output surface is differentiable with respect to
    input vertex positions and the SDF values. For more details and example usage in learning, see
    `Deep Marching Tetrahedra\: a Hybrid Representation for High-Resolution 3D Shape Synthesis`_ NeurIPS 2021.


    Args:
        vertices (torch.Tensor): batched vertices of tetrahedral meshes, of shape
                                 :math:`(\text{batch_size}, \text{num_vertices}, 3)`.
        tets (torch.Tensor): unbatched tetrahedral mesh topology, of shape
                             :math:`(\text{num_tetrahedrons}, 4)`.
        sdf (torch.Tensor): batched SDFs which specify the SDF value of each vertex, of shape
                            :math:`(\text{batch_size}, \text{num_vertices})`.
        return_tet_idx (optional, bool): if True, return index of tetrahedron
                                         where each face is extracted. Default: False.

    Returns:
        (torch.Tensor, torch.LongTensor, (optional) torch.LongTensor):

            - the list of vertices for mesh converted from each tetrahedral grid.
            - the list of faces for mesh converted from each tetrahedral grid.
            - the list of indices that correspond to tetrahedra where faces are extracted.

    Example:
        >>> vertices = torch.tensor([[0, 0, 0],
        ...               [1, 0, 0],
        ...               [0, 1, 0],
        ...               [0, 0, 1]], dtype=torch.float)
        >>> tets = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
        >>> sdf = torch.tensor([-1., -1., 0.5, 0.5], dtype=torch.float)
        >>> verts, faces, tet_idx = marching_tetrahedras(vertices, tets, sdf, True)
        >>> verts
        tensor([[0.0000, 0.6667, 0.0000],
                [0.0000, 0.0000, 0.6667],
                [0.3333, 0.6667, 0.0000],
                [0.3333, 0.0000, 0.6667]])
        >>> faces
        tensor([[3, 0, 1],
                [3, 2, 0]])
        >>> tet_idx
        tensor([0, 0])

    .. _An efficient method of triangulating equi-valued surfaces by using tetrahedral cells:
        https://search.ieice.org/bin/summary.php?id=e74-d_1_214

    .. _Deep Marching Tetrahedra\: a Hybrid Representation for High-Resolution 3D Shape Synthesis:
            https://arxiv.org/abs/2111.04276
    """
    device = vertices.device

    # correct tetrahedras
    flip_mask = _orientation_test(vertices, tets)
    tets[flip_mask, :2] = tets[flip_mask][:, [1, 0]]

    with torch.no_grad():
        occ_n = sdf > 0
        occ_fx4 = occ_n[tets.reshape(-1)].reshape(-1, 4)
        occ_sum = torch.sum(occ_fx4, -1)
        valid_tets = (occ_sum > 0) & (occ_sum < 4)

        # find all edges of all valid tetrahedras and sort
        all_edges = tets[valid_tets][:, base_tet_edges.to(device)].reshape(-1, 2)
        all_edges = _sort_edges(all_edges)
        unique_edges, edge_idx_map = torch.unique(all_edges, dim=0, return_inverse=True)
        unique_edges = unique_edges.long()

        # only exist points when the edge has both positive and negative sdfs
        mask_edges = occ_n[unique_edges].sum(-1) == 1
        mask_mapping = (
            torch.ones((unique_edges.shape[0]), dtype=torch.long, device=device) * -1
        )
        mask_mapping[mask_edges] = torch.arange(
            mask_edges.sum(), dtype=torch.long, device=device
        )  # mark the valid edges
        edge_idx_map = mask_mapping[edge_idx_map]  # find the valid edge ids

        # get the edge's two points to interpolate the vertex
        interp_v = unique_edges[mask_edges]  # [num_vertices, 2]

    """ Find all vertices """
    # interpolate and get the vertices
    edges_to_interp = vertices[interp_v]  # [num_vertices, 2, 3]
    edges_to_interp_sdf = sdf[interp_v]  # [num_vertices, 2]
    # unite the value signs get the denominator to calculate the weights
    edges_to_interp_sdf[:, -1] *= -1
    denominator = edges_to_interp_sdf.sum(1, keepdim=True)  # [num_vertices, 1]

    # calculate the weight and interpolate to get the vertices
    edges_to_interp_sdf = (
        torch.flip(edges_to_interp_sdf, [1]) / denominator
    )  # [num_vertices, 2]
    verts = (edges_to_interp * edges_to_interp_sdf[..., None]).sum(
        1
    )  # [num_vertices, 3]

    """ Find all faces"""
    # get the edges of all invalid tetrahedras (edge ids equal to vertex ids)
    edge_idx_map = edge_idx_map.reshape(-1, 6)  # [num_valid_tets, 6]

    # get the number of triangles and triangle table for each valid tetrahedra
    triangle_table_idx = (occ_fx4[valid_tets] * v_id.to(device).unsqueeze(0)).sum(
        -1
    )  # [num_valid_tets]
    num_triangles = num_triangles_table.to(device)[
        triangle_table_idx
    ]  # [num_valid_tets]
    triangle_table_device = triangle_table.to(device)

    # Generate triangle indices (query the vertex ids using `triangle_table` mapping)
    faces = torch.cat(
        (
            torch.gather(
                input=edge_idx_map[num_triangles == 1],
                dim=1,
                index=triangle_table_device[triangle_table_idx[num_triangles == 1]][
                    :, :3
                ],
            ),
            torch.gather(
                input=edge_idx_map[num_triangles == 2],
                dim=1,
                index=triangle_table_device[triangle_table_idx[num_triangles == 2]][
                    :, :6
                ],
            ).reshape(-1, 3),
        ),
        dim=0,
    )

    if return_tet_idx:
        tet_idx = torch.arange(tets.shape[0], device=device)[valid_tets]
        tet_idx = torch.cat(
            (
                tet_idx[num_triangles == 1],
                tet_idx[num_triangles == 2].repeat_interleave(2),
            ),
            dim=0,
        )
        return verts, faces, tet_idx
    return verts, faces
