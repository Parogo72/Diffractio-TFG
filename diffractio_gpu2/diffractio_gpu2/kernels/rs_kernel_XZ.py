from functools import lru_cache

import numpy as np
from pyvkfft.fft import fftn, ifftn
from ..scalar_fields_X import kernelRS

from ._lazy_cuda import ensure_pycuda_context

_KERNEL_XZ_F32 = """
#include <pycuda-complex.hpp>
#include <math.h>
#define PI 3.14159265358979323846f

__global__ void compute_kernel_RS_XZ_fast_f32(
    pycuda::complex<float>* H,
    const float* xext,
    const float* z_values,
    float k,
    int Nx_ext, int Nz,
    int kind)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= Nx_ext || iz >= Nz) return;

    float X = xext[ix];
    float z = z_values[iz];
    float R = sqrtf(X*X + z*z);

    int idx = iz * Nx_ext + ix;
    if (R < 1e-10f) { H[idx] = pycuda::complex<float>(0.f, 0.f); return; }

    // hk1 = sqrt(2/([pi·k·R)) · exp(i·(kR - 3pi/4))
    float kR      = k * R;
    float hk1_amp = sqrtf(2.0f / (PI * k * R));
    float theta   = kR - 3.0f * PI / 4.0f;
    float hk1_re  = hk1_amp * cosf(theta);
    float hk1_im  = hk1_amp * sinf(theta);

    // (0.5j · factor) · hk1  ->  j·(a+ib) = -b + ia
    float factor;
    if      (kind == 0) factor = k * z / R;
    else if (kind == 1) factor = k * X / R;
    else                factor = k;

    float coeff = 0.5f * factor;
    H[idx] = pycuda::complex<float>(-coeff * hk1_im,
                                     coeff * hk1_re);
}

__global__ void multiply_XZ_f32(
    pycuda::complex<float>* A,
    const pycuda::complex<float>* B,
    int Nx, int Nz)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= Nx || iz >= Nz) return;
    int idx = iz * Nx + ix;
    A[idx] = A[idx] * B[idx];
}
"""

_KERNEL_XZ_F64 = """
#include <pycuda-complex.hpp>
#include <math.h>
#define PI 3.141592653589793238462643383279502884197

__global__ void compute_kernel_RS_XZ_fast_f64(
    pycuda::complex<double>* H,
    const double* xext,
    const double* z_values,
    double k,
    int Nx_ext, int Nz,
    int kind)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= Nx_ext || iz >= Nz) return;

    double X = xext[ix];
    double z = z_values[iz];
    double R = sqrt(X*X + z*z);

    int idx = iz * Nx_ext + ix;
    if (R < 1e-20) { H[idx] = pycuda::complex<double>(0.0, 0.0); return; }

    // hk1 = sqrt(2/(pi·k·R)) · exp(i·(kR - 3pi/4))
    double kR      = k * R;
    double hk1_amp = sqrt(2.0 / (PI * k * R));
    double theta   = kR - 3.0 * PI / 4.0;
    double hk1_re  = hk1_amp * cos(theta);
    double hk1_im  = hk1_amp * sin(theta);

    // (0.5j · factor) · hk1  ->  j·(a+ib) = -b + ia
    double factor;
    if      (kind == 0) factor = k * z / R;
    else if (kind == 1) factor = k * X / R;
    else                factor = k;

    double coeff = 0.5 * factor;
    H[idx] = pycuda::complex<double>(-coeff * hk1_im,
                                      coeff * hk1_re);
}

__global__ void multiply_XZ_f64(
    pycuda::complex<double>* A,
    const pycuda::complex<double>* B,
    int Nx, int Nz)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= Nx || iz >= Nz) return;
    int idx = iz * Nx + ix;
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

    mod_xz_f32 = SourceModule(_KERNEL_XZ_F32)
    mod_xz_f64 = SourceModule(_KERNEL_XZ_F64)

    return {
        'float32': {
            'np_real': np.float32,
            'np_complex': np.complex64,
            'kern_fast': mod_xz_f32.get_function("compute_kernel_RS_XZ_fast_f32"),
            'multiply': mod_xz_f32.get_function("multiply_XZ_f32"),
            'np_k': np.float32,
        },
        'float64': {
            'np_real': np.float64,
            'np_complex': np.complex128,
            'kern_fast': mod_xz_f64.get_function("compute_kernel_RS_XZ_fast_f64"),
            'multiply': mod_xz_f64.get_function("multiply_XZ_f64"),
            'np_k': np.float64,
        },
    }


def _get_cfg(precision):
    return _compiled_kernels()[precision]

def RS_XZ_gpu(u0_field, z_values, x_out=None, n=1.0, kind='z',
              fast=False, precision='float64'):
    if precision not in ('float32', 'float64'):
        raise ValueError(f"precision debe ser 'float32' o 'float64', "
                         f"recibido: '{precision}'")

    cfg        = _get_cfg(precision)
    np_real    = cfg['np_real']
    np_complex = cfg['np_complex']
    kern_fast  = cfg['kern_fast']
    multiply   = cfg['multiply']
    np_k       = cfg['np_k']

    # 1. Datos 
    u0 = u0_field.u.astype(np_complex)
    x  = np.asarray(x_out if x_out is not None else u0_field.x, dtype=np_real)

    Nx    = len(x)
    Nz    = len(z_values)
    z_arr = np.asarray(z_values, dtype=np_real)
    dx    = np_real(x[1] - x[0])
    k     = np_k(2 * np.pi * n / float(u0_field.wavelength))

    # 2. Pesos de Simpson 
    a       = [2, 4]
    num_rep = int(round(Nx / 2) - 1)
    b       = np.array(a * num_rep)
    W       = np.concatenate(([1], b, [2, 1])) / 3.0
    if float(Nx) / 2 == round(Nx / 2):
        i_central = num_rep + 1
        W = np.concatenate((W[:i_central], W[i_central + 1:]))
    W = W.astype(np_real)

    # 3. Grid extendido
    xext   = np.concatenate((x[0] - x[::-1][:-1], x - x[0]))
    Nx_ext = len(xext)

    # 4. Campo con pesos + zero-padding
    U = np.zeros(Nx_ext, dtype=np_complex)
    U[0:Nx] = W * u0

    # 5. FFT del campo una sola vez
    gpuarray = _gpuarray_module()

    U_gpu  = gpuarray.to_gpu(U)
    FU_gpu = fftn(U_gpu)

    # 6. Broadcast para todos los z
    FU_batch = gpuarray.zeros((Nz, Nx_ext), dtype=np_complex)
    for i in range(Nz):
        FU_batch[i] = FU_gpu

    # 7. Kernel RS + FFT del kernel
    block = (32, 32, 1)
    grid  = ((Nx_ext + 31) // 32, (Nz + 31) // 32, 1)

    if fast:
        # Aproximación Hankel — kernel CUDA puro, todo en GPU
        xext_gpu = gpuarray.to_gpu(xext.astype(np_real))
        z_gpu    = gpuarray.to_gpu(z_arr)
        H_gpu    = gpuarray.zeros((Nz, Nx_ext), dtype=np_complex)

        kern_fast(
            H_gpu, xext_gpu, z_gpu, np_k(k),
            np.int32(Nx_ext), np.int32(Nz),
            np.int32(_KIND_MAP[kind]),
            block=block, grid=grid
        )
        FH_gpu = fftn(H_gpu, axes=(-1,))

    else:
        # Hankel exacto — CPU calcula el kernel, GPU hace la FFT batched
        H_cpu = np.zeros((Nz, Nx_ext), dtype=np_complex)
        for iz, z in enumerate(z_arr):
            H_cpu[iz] = kernelRS(
                xext, float(u0_field.wavelength), float(z), n,
                kind=kind, fast=False
            ).astype(np_complex)

        H_gpu  = gpuarray.to_gpu(H_cpu)
        FH_gpu = fftn(H_gpu, axes=(-1,))

    # 8. Multiplicación espectral
    multiply(
        FU_batch, FH_gpu,
        np.int32(Nx_ext), np.int32(Nz),
        block=block, grid=grid
    )

    # 9. IFFT batched + escalado
    S_gpu = ifftn(FU_batch, axes=(-1,))
    S     = S_gpu.get() * float(dx)

    # 10. Extraer región válida
    return S[:, Nx - 1:]