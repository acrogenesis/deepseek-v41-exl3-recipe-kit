"""Opt-in native dispatch linked against the pinned ExLlama build."""
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def load_dispatch():
    from torch.utils.cpp_extension import load
    import exllamav3_ext
    build = Path('/kernel-cache/exact-dispatch')
    build.mkdir(parents=True, exist_ok=True)
    native = Path(exllamav3_ext.__file__)
    return load(name='deepseek41_exact_dispatch',
        sources=[str(Path(__file__).with_suffix('.cpp')), str(Path(__file__).with_name('grouped_dispatch.cu'))],
        extra_include_paths=['/opt/deepseek-exllamav3/exllamav3/exllamav3_ext'],
        extra_cflags=['-O3'], extra_cuda_cflags=['-O3', '-lineinfo', '--use_fast_math'],
        extra_ldflags=[str(native), '-Wl,-rpath,' + str(native.parent)],
        build_directory=str(build), with_cuda=True, verbose=True)
