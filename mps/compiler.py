import torch.utils.cpp_extension

def compile_and_load_kernel(name, sources):
    """
    JIT-compiles and loads a custom kernel extension.
    """
    return torch.utils.cpp_extension.load(
        name=name,
        sources=sources,
        extra_cflags=['-std=c++17', '-fobjc-arc', '-arch arm64'],
        extra_ldflags=['-arch arm64'],
        verbose=True,
    )
