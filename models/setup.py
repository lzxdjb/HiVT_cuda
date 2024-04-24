from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='graph',
    ext_modules=[
        CUDAExtension('graph', [
            'graph.cu'
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })