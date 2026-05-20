# !/usr/bin/env python3
"""
Helpers simples para transferencia de datos GPU/CPU.
"""

import numpy as np

try:
    import pycuda.driver as cuda
    _CUDA_AVAILABLE = True
except ImportError:
    cuda = None
    _CUDA_AVAILABLE = False


def copy_to_gpu(arr: np.ndarray, dtype=np.complex64) -> object:
    """
    Copia array a GPU.
    
    Args:
        arr: Array NumPy
        dtype: Tipo de dato en GPU
    
    Returns:
        Device pointer (int) o None si PyCUDA no disponible
    """
    if not _CUDA_AVAILABLE:
        return None
    
    try:
        if arr.dtype != dtype:
            arr = arr.astype(dtype)
        d_arr = cuda.mem_alloc(arr.nbytes)
        cuda.memcpy_htod(d_arr, arr)
        return d_arr
    except Exception as e:
        print(f"Error transferring to GPU: {e}")
        return None


def copy_from_gpu(d_arr, shape: tuple, dtype=np.complex64) -> np.ndarray:
    """
    Copia array desde GPU a CPU.
    
    Args:
        d_arr: Device pointer
        shape: Shape del array
        dtype: Tipo de dato
    
    Returns:
        Array NumPy o None si error
    """
    if not _CUDA_AVAILABLE or d_arr is None:
        return None
    
    try:
        arr = np.empty(shape, dtype=dtype)
        cuda.memcpy_dtoh(arr, d_arr)
        return arr
    except Exception as e:
        print(f"Error transferring from GPU: {e}")
        return None


def free_gpu_memory(d_ptr):
    """Libera memoria GPU"""
    if not _CUDA_AVAILABLE or d_ptr is None:
        return
    
    try:
        d_ptr.free()
    except:
        pass
