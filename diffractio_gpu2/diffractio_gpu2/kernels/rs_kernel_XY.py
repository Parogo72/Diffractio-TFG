from functools import lru_cache

import numpy as np
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
from pyvkfft.fft import fftn, ifftn

from ._lazy_cuda import ensure_pycuda_context

_KERNEL_XY_F32 = """
#include <pycuda-complex.hpp>
#include <math.h>
#define PI 3.14159265358979323846f

__global__ void compute_kernel_RS_XY_f32(
    pycuda::complex<float>* H,
    const float* xext,
    const float* yext,
    float z, float k,
    int Nx_ext, int Ny_ext, int kind)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= Nx_ext || iy >= Ny_ext) return;

    float X = xext[ix];
    float Y = yext[iy];
    float R = sqrtf(X*X + Y*Y + z*z);
    int idx = iy * Nx_ext + ix;
    if (R < 1e-10f) { H[idx] = pycuda::complex<float>(0.f, 0.f); return; }

    float kR    = k * R;
    float inv_R = 1.0f / R;
    float real_part = cosf(kR) * inv_R + k * sinf(kR);
    float imag_part = sinf(kR) * inv_R - k * cosf(kR);

    float prefactor;
    if      (kind == 0) prefactor = z     / (2.0f * PI * R * R);
    else if (kind == 1) prefactor = X     / (2.0f * PI * R * R);
    else if (kind == 2) prefactor = Y     / (2.0f * PI * R * R);
    else                prefactor = inv_R / (2.0f * PI);

    H[idx] = pycuda::complex<float>(prefactor * real_part,
                                    prefactor * imag_part);
}

__global__ void multiply_XY_f32(
    pycuda::complex<float>* A,
    const pycuda::complex<float>* B,
    int Nx, int Ny)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= Nx || iy >= Ny) return;
    int idx = iy * Nx + ix;
    A[idx] = A[idx] * B[idx];
}
"""

_KERNEL_XY_F64 = """
#include <pycuda-complex.hpp>
#include <math.h>
#define PI 3.141592653589793238462643383279502884197

__global__ void compute_kernel_RS_XY_f64(
    pycuda::complex<double>* H,
    const double* xext,
    const double* yext,
    double z, double k,
    int Nx_ext, int Ny_ext, int kind)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= Nx_ext || iy >= Ny_ext) return;

    double X = xext[ix];
    double Y = yext[iy];
    double R = sqrt(X*X + Y*Y + z*z);
    int idx = iy * Nx_ext + ix;
    if (R < 1e-20) { H[idx] = pycuda::complex<double>(0.0, 0.0); return; }

    double kR    = k * R;
    double inv_R = 1.0 / R;
    double real_part = cos(kR) * inv_R + k * sin(kR);
    double imag_part = sin(kR) * inv_R - k * cos(kR);

    double prefactor;
    if      (kind == 0) prefactor = z     / (2.0 * PI * R * R);
    else if (kind == 1) prefactor = X     / (2.0 * PI * R * R);
    else if (kind == 2) prefactor = Y     / (2.0 * PI * R * R);
    else                prefactor = inv_R / (2.0 * PI);

    H[idx] = pycuda::complex<double>(prefactor * real_part,
                                     prefactor * imag_part);
}

__global__ void multiply_XY_f64(
    pycuda::complex<double>* A,
    const pycuda::complex<double>* B,
    int Nx, int Ny)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= Nx || iy >= Ny) return;
    int idx = iy * Nx + ix;
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

    mod_xy_f32 = SourceModule(_KERNEL_XY_F32)
    mod_xy_f64 = SourceModule(_KERNEL_XY_F64)

    return {
        'float32': {
            'np_real':    np.float32,
            'np_complex': np.complex64,
            'kern_rs':    mod_xy_f32.get_function("compute_kernel_RS_XY_f32"),
            'multiply':   mod_xy_f32.get_function("multiply_XY_f32"),
            'np_k':       np.float32,
        },
        'float64': {
            'np_real':    np.float64,
            'np_complex': np.complex128,
            'kern_rs':    mod_xy_f64.get_function("compute_kernel_RS_XY_f64"),
            'multiply':   mod_xy_f64.get_function("multiply_XY_f64"),
            'np_k':       np.float64,
        },
    }


def _get_cfg(precision):
    return _compiled_kernels()[precision]


def _RS_XY_gpu(u0_field, z, n=1.0, kind='z',
               xout=None, yout=None, precision='float32'):
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
    z_val  = np_real(z)
    dx     = np_real(x[1] - x[0])
    dy     = np_real(y[1] - y[0])
    k      = np_k(2 * np.pi * n / float(u0_field.wavelength))

    # 2. xout, yout 
    xout_val = np_real(xout) if xout is not None else x[0]
    yout_val = np_real(yout) if yout is not None else y[0]
    xout_arr = x + xout_val - x[0]
    yout_arr = y + yout_val - y[0]

    # 3. Grid extendido 
    xext   = np.concatenate((x[0] - xout_arr[::-1][:-1], x - xout_arr[0]))
    yext   = np.concatenate((y[0] - yout_arr[::-1][:-1], y - yout_arr[0]))
    Nx_ext = len(xext)
    Ny_ext = len(yext)

    # 4. Zero-padding 
    U_padded = np.zeros((Ny_ext, Nx_ext), dtype=np_complex)
    U_padded[0:Ny, 0:Nx] = u0

    try:
        gpuarray = _gpuarray_module()

        # 5. FFT del campo 
        U_gpu  = gpuarray.to_gpu(U_padded)
        FU_gpu = fftn(U_gpu)
        del U_gpu                              # liberar U_gpu, ya no hace falta

        # 6. Kernel RS 
        xext_gpu = gpuarray.to_gpu(xext.astype(np_real))
        yext_gpu = gpuarray.to_gpu(yext.astype(np_real))
        H_gpu    = gpuarray.zeros((Ny_ext, Nx_ext), dtype=np_complex)

        block = (16, 16, 1)
        grid  = ((Nx_ext + 15) // 16, (Ny_ext + 15) // 16, 1)

        kern_rs(
            H_gpu, xext_gpu, yext_gpu,
            np_real(z_val), np_k(k),
            np.int32(Nx_ext), np.int32(Ny_ext),
            np.int32(_KIND_MAP[kind]),
            block=block, grid=grid
        )
        del xext_gpu, yext_gpu                 # liberar coords

        # 7. FFT del kernel
        FH_gpu = fftn(H_gpu)
        del H_gpu                              # liberar H_gpu

        # 8. Multiplicación
        FU_copy = FU_gpu.copy()
        multiply(
            FU_copy, FH_gpu,
            np.int32(Nx_ext), np.int32(Ny_ext),
            block=block, grid=grid
        )
        del FU_gpu, FH_gpu                

        # 9. IFFT + escalado 
        S_gpu = ifftn(FU_copy)
        del FU_copy

        S = S_gpu.get() * float(dx) * float(dy)
        del S_gpu

        # 10. Extraer región válida
        return S[Ny - 1:, Nx - 1:]

    except cuda.MemoryError as e:
        # Limpiar contexto GPU si hay OOM
        cuda.Context.synchronize()
        raise RuntimeError(
            f"GPU OOM en _RS_XY_gpu: grid ({Ny_ext}×{Nx_ext}) "
            f"precision={precision}. "
            f"Prueba precision='float32'."
        ) from e


def RS_XY_gpu(u0_field, z, amplification=(1, 1), n=1.0, kind='z',
              xout=None, yout=None, precision='float64'):
    """
    Equivalente GPU de Scalar_field_XY.RS().
    """
    cfg        = _get_cfg(precision)
    np_real    = cfg['np_real']
    np_complex = cfg['np_complex']

    x = u0_field.x.astype(np_real)
    y = u0_field.y.astype(np_real)

    amp_x, amp_y = amplification
    width_x      = x[-1] - x[0]
    width_y      = y[-1] - y[0]
    num_pixels_x = len(x)
    num_pixels_y = len(y)

    # Caso amplification = (1,1)
    if amp_x * amp_y == 1:
        xout_rs = None if xout is None else -xout + x[0] - width_x / 2
        yout_rs = None if yout is None else -yout + y[0] - width_y / 2

        U = _RS_XY_gpu(u0_field, z, n=n, kind=kind,
                       xout=xout_rs, yout=yout_rs, precision=precision)
        X0 = x.copy() if xout is None else x + xout - x[0]
        Y0 = y.copy() if yout is None else y + yout - y[0]
        return X0, Y0, U

    # Caso amplification > (1,1) 
    posiciones_x = (
        -amp_x * width_x / 2 +
        np.arange(amp_x, dtype=np_real) * width_x
    )
    posiciones_y = (
        -amp_y * width_y / 2 +
        np.arange(amp_y, dtype=np_real) * width_y
    )

    X0 = np.linspace(-amp_x * width_x / 2,  amp_x * width_x / 2,
                     num_pixels_x * amp_x, dtype=np_real)
    Y0 = np.linspace(-amp_y * width_y / 2,  amp_y * width_y / 2,
                     num_pixels_y * amp_y, dtype=np_real)

    U_final = np.zeros((num_pixels_y * amp_y, num_pixels_x * amp_x),
                       dtype=np_complex)

    for i, xi in zip(range(amp_x), np.flipud(posiciones_x)):
        for j, yi in zip(range(amp_y), np.flipud(posiciones_y)):

            u_frame = _RS_XY_gpu(
                u0_field, z, n=n, kind=kind,
                xout=xi, yout=yi, precision=precision
            )

            xshape = slice(i * num_pixels_x, (i + 1) * num_pixels_x)
            yshape = slice(j * num_pixels_y, (j + 1) * num_pixels_y)
            U_final[yshape, xshape] = u_frame

            # Sincronizar y liberar memoria GPU entre frames
            cuda.Context.synchronize()

    return X0, Y0, U_final