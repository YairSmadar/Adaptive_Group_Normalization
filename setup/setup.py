from setuptools import setup, Extension
from torch.utils import cpp_extension


extension = cpp_extension.CppExtension(
    name='AGN',
    sources=['AGN.cpp'],
    include_dirs=cpp_extension.include_paths(),
    libraries=["c10", "torch", "torch_cpu"],
    library_dirs=["C:\\Users\\yair.semama\\AppData\\Local\\LibTorch\\libtorch-win-shared-with-deps-1.13.1+cpu\\libtorch\\lib"],
    language='c++',
)

setup(
    name='AGN',
    ext_modules=[extension],
    cmdclass={'build_ext': cpp_extension.BuildExtension},
)
