# Primitive3D

Primitive3D: Self-Pratice Library for pytorch-based 3D Data Processing.

# Installation
```sh
export OptiX_INSTALL_DIR=$OptiX_INSTALL_DIR # (optional)
python setup.py develop
```

## Examples

- Ray Casting
```python
import torch
import prim3d

vertices: torch.Tensor # input vertices
faces: torch.Tensor # input faces
# create the ray caster
ray_caster = prim3d.create_raycaster(vertices, faces)

# setting ray origins and directions
origins: torch.Tensor
dirs: torch.Tensor

# output contained
depths: torch.Tensor
normals: torch.Tensor
primitive_ids: torch.Tensor

# conduct ray casting
ray_caster.invoke(origins, dirs, depths, normals, primitive_ids)
```

- Marching Cubes
```sh
# toy examples from the PyMCubes
python examples/sphere.py
# we extract the sdf of bunny via `mesh_to_sdf`
python examples/bunny_sdf.py
# example for extracting a sphere using marching tetrahedra
python examples/sphere_tetrahedra.py
```

## TODO
- [x] RayCasting based on Optix or CUDA BVH
- [x] Marching Cubes(Mitigate from [CuMCubes](https://github.com/lzhnb/CuMCubes))
- [ ] Add RayCasting Examples
- [ ] SDF From Mesh
- [ ] Documents


## Acknowledgement
- [Open3D](https://github.com/isl-org/Open3D)
- [PyMesh](https://github.com/PyMesh/PyMesh)
- [instant-npg](https://github.com/NVlabs/instant-ngp)
- [AnalyticMesh](https://github.com/Gorilla-Lab-SCUT/AnalyticMesh)

## ResourcesPermalink
- [How to get started with OptiX 7](https://developer.nvidia.com/blog/how-to-get-started-with-optix-7/)

> Please feel free to discuss :)
