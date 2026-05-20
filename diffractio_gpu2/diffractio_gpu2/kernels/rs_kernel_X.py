from functools import lru_cache

import numpy as np
from pyvkfft.fft import fftn, ifftn

from ._lazy_cuda import ensure_pycuda_context

_KERNEL_X_F32 = """
#include <pycuda-complex.hpp>
#include <math.h>
#define PI 3.14159265358979323846f

__global__ void compute_kernel_RS_X_f32(
    pycuda::complex<float>* H,    // salida: (Nx_ext,)
    const float* xext,            // coordenadas x extendidas: (Nx_ext,)
    float z,                      // distancia escalar
    float k,
    int Nx_ext,
    int kind)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    if (ix >= Nx_ext) return;

    float X = xext[ix];
    // En X no hay Y: R = sqrt(X² + z²)
    float R = sqrtf(X*X + z*z);

    if (R < 1e-10f) {
        H[ix] = pycuda::complex<float>(0.f, 0.f);
        return;
    }

    float kR    = k * R;
    float inv_R = 1.0f / R;
    float real_part = cosf(kR) * inv_R + k * sinf(kR);
    float imag_part = sinf(kR) * inv_R - k * cosf(kR);

    float prefactor;
    if      (kind == 0) prefactor = z     / (2.0f * PI * R * R);  // 'z'
    else if (kind == 1) prefactor = X     / (2.0f * PI * R * R);  // 'x'
    else                prefactor = inv_R / (2.0f * PI);           // '0'

    H[ix] = pycuda::complex<float>(prefactor * real_part,
                                   prefactor * imag_part);
}

__global__ void multiply_X_f32(
    pycuda::complex<float>* A,
    const pycuda::complex<float>* B,
    int Nx)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    if (ix >= Nx) return;
    A[ix] = A[ix] * B[ix];
}
"""

_KERNEL_X_F64 = """
#include <pycuda-complex.hpp>
#include <math.h>
#define PI 3.141592653589793238462643383279502884197

__global__ void compute_kernel_RS_X_f64(
    pycuda::complex<double>* H,
    const double* xext,
    double z,
    double k,
    int Nx_ext,
    int kind)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    if (ix >= Nx_ext) return;

    double X = xext[ix];
    double R = sqrt(X*X + z*z);

    if (R < 1e-20) {
        H[ix] = pycuda::complex<double>(0.0, 0.0);
        return;
    }

    double kR    = k * R;
    double inv_R = 1.0 / R;
    double real_part = cos(kR) * inv_R + k * sin(kR);
    double imag_part = sin(kR) * inv_R - k * cos(kR);

    double prefactor;
    if      (kind == 0) prefactor = z     / (2.0 * PI * R * R);
    else if (kind == 1) prefactor = X     / (2.0 * PI * R * R);
    else                prefactor = inv_R / (2.0 * PI);

    H[ix] = pycuda::complex<double>(prefactor * real_part,
                                    prefactor * imag_part);
}

__global__ void multiply_X_f64(
    pycuda::complex<double>* A,
    const pycuda::complex<double>* B,
    int Nx)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    if (ix >= Nx) return;
    A[ix] = A[ix] * B[ix];
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

    mod_x_f32 = SourceModule(_KERNEL_X_F32)
    mod_x_f64 = SourceModule(_KERNEL_X_F64)

    return {
        'float32': {
            'np_real': np.float32,
            'np_complex': np.complex64,
            'kern_rs': mod_x_f32.get_function("compute_kernel_RS_X_f32"),
            'multiply': mod_x_f32.get_function("multiply_X_f32"),
            'np_k': np.float32,
        },
        'float64': {
            'np_real': np.float64,
            'np_complex': np.complex128,
            'kern_rs': mod_x_f64.get_function("compute_kernel_RS_X_f64"),
            'multiply': mod_x_f64.get_function("multiply_X_f64"),
            'np_k': np.float64,
        },
    }


def _get_cfg(precision):
    return _compiled_kernels()[precision]


def _RS_X_gpu(u0_field, z, n=1.0, kind='z', xout=None, precision='float64'):
    """
    Equivalente GPU de Scalar_field_X._RS_() para un solo z.

    Args:
        u0_field  : Scalar_field_X
        z         : distancia de propagación (float)
        n         : índice de refracción
        kind      : 'z', 'x', '0'
        xout      : posición inicial del plano de salida (None = x[0])
        precision : 'float32' | 'float64'

    Returns:
        np.ndarray shape (Nx,) — equivalente a Usalida en _RS_
    """
    cfg        = _get_cfg(precision)
    np_real    = cfg['np_real']
    np_complex = cfg['np_complex']
    kern_rs    = cfg['kern_rs']
    multiply   = cfg['multiply']
    np_k       = cfg['np_k']

    # 1. Datos del campo
    u0 = u0_field.u.astype(np_complex)
    x  = u0_field.x.astype(np_real)
    Nx = len(x)
    dx = np_real(x[1] - x[0])
    k  = np_k(2 * np.pi * n / float(u0_field.wavelength))
    z  = np_real(z)

    # 2. xout
    if xout is None:
        xout_val = x[0]
    else:
        xout_val = np_real(xout)

    xout_arr = x + xout_val - x[0]  

    # 3. Grid extendido 
    xext = np.concatenate((x[0] - xout_arr[::-1][:-1], x - xout_arr[0]))
    Nx_ext = len(xext)

    # 4. Zero-padding del campo (W=1, precise=False)
    U = np.zeros(Nx_ext, dtype=np_complex)
    U[0:Nx] = u0

    # 5. Subir a GPU y FFT del campo
    gpuarray = _gpuarray_module()

    U_gpu  = gpuarray.to_gpu(U)
    FU_gpu = fftn(U_gpu) 

    # 6. Kernel RS en GPU (z escalar, 1D) 
    xext_gpu = gpuarray.to_gpu(xext.astype(np_real))
    H_gpu    = gpuarray.zeros(Nx_ext, dtype=np_complex)

    block = (256, 1, 1)
    grid  = ((Nx_ext + 255) // 256, 1, 1)

    kern_rs(
        H_gpu,
        xext_gpu,
        np_real(z),
        np_k(k),
        np.int32(Nx_ext),
        np.int32(_KIND_MAP[kind]),
        block=block, grid=grid
    )

    # 7. FFT del kernel RS
    FH_gpu = fftn(H_gpu)

    # 8. Multiplicación espectral
    multiply(
        FU_gpu, FH_gpu, 
        np.int32(Nx_ext),
        block=block, grid=grid
    )

    # 9. IFFT + escalado 
    S_gpu = ifftn(FU_gpu)
    S     = S_gpu.get() * float(dx)

    # 10. Extraer región válida
    return S[Nx - 1:]                 # → (Nx,)

def RS_X_gpu(u0_field, z, amplification=1, n=1.0, kind='z',
             xout=None, precision='float64'):
    """
    Equivalente GPU de Scalar_field_X.RS().
    Itera sobre frames de amplification, cada uno procesado en GPU.

    Args:
        u0_field      : Scalar_field_X
        z             : distancia de propagación (float)
        amplification : número de frames en x (default 1)
        n             : índice de refracción
        kind          : 'z', 'x', '0'
        xout          : desplazamiento del plano de salida
        precision     : 'float32' | 'float64'

    Returns:
        np.ndarray shape (Nx * amplification,)
    """
    cfg = _get_cfg(precision)
    np_real = cfg['np_real']

    x          = u0_field.x.astype(np_real)
    width_x    = x[-1] - x[0]
    num_pixels = len(x)

    # Posiciones de los frames
    positions_x = (
        -amplification * width_x / 2 +
        np.arange(amplification, dtype=np_real) * width_x
    )

    if xout is not None:
        positions_x = positions_x + np_real(xout)

    # Grid de salida completo
    x0 = np.linspace(
        -amplification * width_x / 2,
        amplification * width_x / 2,
        num_pixels * amplification,
        dtype=np_real
    )
    if xout is not None:
        x0 = x0 + np_real(xout) - width_x / 2

    # Iterar sobre frames
    u_out = np.zeros(num_pixels * amplification,
                     dtype=cfg['np_complex'])

    for i, xi in zip(range(amplification), np.flipud(positions_x)):
        u_frame = _RS_X_gpu(
            u0_field  = u0_field,
            z         = z,
            n         = n,
            kind      = kind,
            xout      = xi,
            precision = precision
        )
        xshape = slice(i * num_pixels, (i + 1) * num_pixels)
        u_out[xshape] = u_frame 

    return x0, u_out