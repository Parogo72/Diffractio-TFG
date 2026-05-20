from functools import lru_cache

import numpy as np
from pyvkfft.fft import fftn, ifftn

from ._lazy_cuda import ensure_pycuda_context

_KERNEL_CODE_F32 = """
#include <pycuda-complex.hpp>
#include <math.h>
#define PI 3.14159265358979323846f

__global__ void compute_kernel_RS_f32(
    pycuda::complex<float>* H,
    const float* xext,
    const float* yext,
    const float* z_values,
    float k,
    int Nx_ext, int Ny_ext, int Nz,
    int kind)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;
    if (ix >= Nx_ext || iy >= Ny_ext || iz >= Nz) return;

    float X = xext[ix];
    float Y = yext[iy];
    float z = z_values[iz];
    float R = sqrtf(X*X + Y*Y + z*z);

    int idx = iz * Nx_ext * Ny_ext + iy * Nx_ext + ix;
    if (R < 1e-10f) { H[idx] = pycuda::complex<float>(0.f, 0.f); return; }

    float kR     = k * R;
    float inv_R  = 1.0f / R;
    float real_part = cosf(kR) * inv_R + k * sinf(kR);
    float imag_part = sinf(kR) * inv_R - k * cosf(kR);

    float prefactor;
    if      (kind == 0) prefactor = z    / (2.0f * PI * R * R);
    else if (kind == 1) prefactor = X    / (2.0f * PI * R * R);
    else if (kind == 2) prefactor = Y    / (2.0f * PI * R * R);
    else                prefactor = inv_R / (2.0f * PI);

    H[idx] = pycuda::complex<float>(prefactor * real_part,
                                    prefactor * imag_part);
}

__global__ void multiply_f32(
    pycuda::complex<float>* A,
    const pycuda::complex<float>* B,
    int Nx, int Ny, int Nz)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;
    if (ix >= Nx || iy >= Ny || iz >= Nz) return;
    int idx = iz * Nx * Ny + iy * Nx + ix;
    A[idx] = A[idx] * B[idx];
}
"""

_KERNEL_CODE_F64 = """
#include <pycuda-complex.hpp>
#include <math.h>
#define PI 3.141592653589793238462643383279502884197

__global__ void compute_kernel_RS_f64(
    pycuda::complex<double>* H,
    const double* xext,
    const double* yext,
    const double* z_values,
    double k,
    int Nx_ext, int Ny_ext, int Nz,
    int kind)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;
    if (ix >= Nx_ext || iy >= Ny_ext || iz >= Nz) return;

    double X = xext[ix];
    double Y = yext[iy];
    double z = z_values[iz];
    double R = sqrt(X*X + Y*Y + z*z);

    int idx = iz * Nx_ext * Ny_ext + iy * Nx_ext + ix;
    if (R < 1e-20) { H[idx] = pycuda::complex<double>(0.0, 0.0); return; }

    double kR     = k * R;
    double inv_R  = 1.0 / R;
    double real_part = cos(kR) * inv_R + k * sin(kR);
    double imag_part = sin(kR) * inv_R - k * cos(kR);

    double prefactor;
    if      (kind == 0) prefactor = z    / (2.0 * PI * R * R);
    else if (kind == 1) prefactor = X    / (2.0 * PI * R * R);
    else if (kind == 2) prefactor = Y    / (2.0 * PI * R * R);
    else                prefactor = inv_R / (2.0 * PI);

    H[idx] = pycuda::complex<double>(prefactor * real_part,
                                     prefactor * imag_part);
}

__global__ void multiply_f64(
    pycuda::complex<double>* A,
    const pycuda::complex<double>* B,
    int Nx, int Ny, int Nz)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;
    if (ix >= Nx || iy >= Ny || iz >= Nz) return;
    int idx = iz * Nx * Ny + iy * Nx + ix;
    A[idx] = A[idx] * B[idx];
}
"""

_KIND_MAP = {'z': 0, 'x': 1, 'y': 2, '0': 3}

@lru_cache(maxsize=1)
def _gpuarray_module():
    ensure_pycuda_context()
    import pycuda.gpuarray as gpuarray
    return gpuarray


@lru_cache(maxsize=1)
def _compiled_kernels():
    ensure_pycuda_context()
    from pycuda.compiler import SourceModule

    mod_f32 = SourceModule(_KERNEL_CODE_F32)
    mod_f64 = SourceModule(_KERNEL_CODE_F64)

    return {
        'float32': {
            'np_real': np.float32,
            'np_complex': np.complex64,
            'kern_rs': mod_f32.get_function("compute_kernel_RS_f32"),
            'multiply': mod_f32.get_function("multiply_f32"),
            'np_k': np.float32,
        },
        'float64': {
            'np_real': np.float64,
            'np_complex': np.complex128,
            'kern_rs': mod_f64.get_function("compute_kernel_RS_f64"),
            'multiply': mod_f64.get_function("multiply_f64"),
            'np_k': np.float64,
        },
    }


def _get_cfg(precision):
    return _compiled_kernels()[precision]

def RS_XYZ_gpu(u0_field, z_values, n=1.0, kind='z', precision='float64', chunk_size=None):
    if precision not in ('float32', 'float64'):
        raise ValueError(f"precision debe ser 'float32' o 'float64', "
                         f"recibido: '{precision}'")

    cfg        = _get_cfg(precision)
    np_real    = cfg['np_real']
    np_complex = cfg['np_complex']
    kern_rs    = cfg['kern_rs']
    multiply   = cfg['multiply']
    np_k       = cfg['np_k']

    # 1. Datos del campo
    u0 = u0_field.u.astype(np_complex)
    x  = u0_field.x.astype(np_real)
    y  = u0_field.y.astype(np_real)

    Ny, Nx = u0.shape
    Nz     = len(z_values)
    z_arr  = np.asarray(z_values, dtype=np_real)

    dx = np_real(x[1] - x[0])
    dy = np_real(y[1] - y[0])
    k  = np_k(2 * np.pi * n / float(u0_field.wavelength))

    # 2. Grid extendido
    xout   = x
    yout   = y
    xext   = np.concatenate((x[0] - xout[::-1][:-1], x - xout[0]))
    yext   = np.concatenate((y[0] - yout[::-1][:-1], y - yout[0]))
    Nx_ext = len(xext) 
    Ny_ext = len(yext)   

    # 3. Auto-detectar chunk_size según VRAM disponible
    if chunk_size is None:
        import pycuda.driver as cuda
        free_mem, total_mem = cuda.mem_get_info()
        # Cada chunk necesita ~3 arrays (FU_batch, H_gpu, FH_gpu)
        # de shape (chunk, Ny_ext, Nx_ext)
        bytes_per_elem  = np.dtype(np_complex).itemsize
        bytes_per_plane = Ny_ext * Nx_ext * bytes_per_elem
        # Usar 70% de la memoria libre para dejar margen
        chunk_size = max(1, int(free_mem * 0.70 / (3 * bytes_per_plane)))
        chunk_size = min(chunk_size, Nz)  # no más que Nz

    # 4. Zero-padding + FFT del campo (una sola vez, fuera del chunk loop)
    U_padded = np.zeros((Ny_ext, Nx_ext), dtype=np_complex)
    U_padded[0:Ny, 0:Nx] = u0

    gpuarray = _gpuarray_module()

    U_gpu  = gpuarray.to_gpu(U_padded)
    FU_gpu = fftn(U_gpu)                   # (Ny_ext, Nx_ext) — reutilizado
    del U_gpu
    
    # Arrays de resultado en CPU
    xext_gpu = gpuarray.to_gpu(xext.astype(np_real))
    yext_gpu = gpuarray.to_gpu(yext.astype(np_real))

    result = np.zeros((Nz, Ny, Nx), dtype=np_complex)

    # ── 5. Loop por chunks de z ───────────────────────────────────────────────
    for z_start in range(0, Nz, chunk_size):
        z_end  = min(z_start + chunk_size, Nz)
        nz_c   = z_end - z_start              # tamaño del chunk actual
        z_chunk = z_arr[z_start:z_end]

        # Broadcast FU para este chunk
        FU_batch = gpuarray.zeros((nz_c, Ny_ext, Nx_ext), dtype=np_complex)
        for i in range(nz_c):
            FU_batch[i] = FU_gpu

        # Kernel RS para este chunk
        z_gpu = gpuarray.to_gpu(z_chunk)
        H_gpu = gpuarray.zeros((nz_c, Ny_ext, Nx_ext), dtype=np_complex)

        block = (16, 16, 4)
        grid  = (
            (Nx_ext + 15) // 16,
            (Ny_ext + 15) // 16,
            (nz_c   +  3) //  4,
        )

        kern_rs(
            H_gpu,
            xext_gpu, yext_gpu, z_gpu,
            np_k(k),
            np.int32(Nx_ext), np.int32(Ny_ext), np.int32(nz_c),
            np.int32(_KIND_MAP[kind]),
            block=block, grid=grid
        )

        # FFT del kernel + multiplicación + IFFT
        FH_gpu = fftn(H_gpu, axes=(-2, -1))

        multiply(
            FU_batch, FH_gpu,
            np.int32(Nx_ext), np.int32(Ny_ext), np.int32(nz_c),
            block=block, grid=grid
        )

        S_gpu = ifftn(FU_batch, axes=(-2, -1))
        S     = S_gpu.get() * float(dx) * float(dy)  # (nz_c, Ny_ext, Nx_ext)

        # Extraer región válida y guardar en resultado
        result[z_start:z_end] = S[:, Ny-1:, Nx-1:]

        # Liberar memoria GPU del chunk explícitamente
        del FU_batch, H_gpu, FH_gpu, S_gpu, z_gpu

    return np.moveaxis(result, 0, -1)