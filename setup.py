import os
import platform
import re
import subprocess
import sys
from pathlib import Path
from sysconfig import get_paths

import torch
from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext

__version__ = None
exec(open("prim3d/version.py", "r").read())
LIBTORCH_ROOT = str(Path(torch.__file__).parent)
PYTHON_VERSION = f"{sys.version_info.major}.{sys.version_info.minor}"


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir="", library_dirs=[]):
        Extension.__init__(self, name, sources=[], library_dirs=library_dirs)
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: "
                + ", ".join(e.name for e in self.extensions)
            )

        if platform.system() == "Windows":
            from distutils.version import LooseVersion

            cmake_version = LooseVersion(re.search(r"version\s*([\d.]+)", out.decode()).group(1))
            if cmake_version < "3.13.0":
                raise RuntimeError("CMake >= 3.13.0 is required on Windows")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = [
            f"-DCMAKE_PREFIX_PATH={torch.utils.cmake_prefix_path}",
            f"-DPYBIND11_PYTHON_VERSION={PYTHON_VERSION}",
            f"-DPYTHON_INCLUDE_DIR={get_paths()['include']}",
            f'-DCMAKE_CUDA_FLAGS="--expt-relaxed-constexpr"',
        ]
        cfg = "Debug" if self.debug else "Release"
        assert cfg == "Release", "pytorch ops don't support debug build."
        build_args = ["--config", cfg]
        print(f"build config: {cfg}")
        if platform.system() == "Windows":
            cmake_args += ["-DCMAKE_BUILD_TYPE=" + cfg]
            cmake_args += [
                f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{cfg.upper()}={str(Path(extdir) / 'prim3d')}"
            ]
            cmake_args += [
                f"-DCMAKE_RUNTIME_OUTPUT_DIRECTORY_{cfg.upper()}={str(Path(extdir) / 'prim3d')}"
            ]
            if sys.maxsize > 2**32:
                cmake_args += ["-A", "x64"]
            build_args += ["--", "/m"]
        else:
            cmake_args += [f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={str(Path(extdir) / 'prim3d')}"]
            cmake_args += ["-DCMAKE_BUILD_TYPE=" + cfg]
            build_args += ["--", "-j16"]

        env = os.environ.copy()
        env[
            "CXXFLAGS"
        ] = f'{env.get("CXXFLAGS", "")} -DVERSION_INFO=\\"{self.distribution.get_version()}\\"'
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        print("CMAKE ARGS:\n", cmake_args)
        subprocess.check_call(["cmake", ext.sourcedir, "-B", "build"] + cmake_args, cwd=self.build_temp, env=env)
        subprocess.check_call(["cmake", "--build", "build"] + build_args, cwd=self.build_temp)


setup(
    name="prim3d",
    version=__version__,
    author="Zhihao Liang",
    author_email="eezhihaoliang@mail.scut.edu.cn",
    description="Primitive3D: Self-Pratice Library for 3D Data Processing.",
    long_description="Primitive3D: Self-Pratice Library for 3D Data Processing.",
    packages=find_packages(),
    package_dir={"prim3d": "prim3d"},
    ext_modules=[CMakeExtension(name="prim3d", library_dirs=[])],
    cmdclass={"build_ext": CMakeBuild},
)
