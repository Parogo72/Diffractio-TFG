import time, numpy as np
from diffractio_gpu2.diffractio_gpu2.scalar_fields_XY import Scalar_field_XY
from diffractio_gpu2.diffractio_gpu2.kernels.rs_kernel_XY import RS_XY_gpu
import pycuda.driver as cuda

x = np.linspace(-100,100,256)
y = np.linspace(-100,100,256)
u0 = Scalar_field_XY(x=x, y=y, wavelength=0.5)
u0.u = np.ones((256,256), dtype=np.complex64)

cuda.Context.synchronize()
t0 = time.perf_counter()
n = 3
for _ in range(n):
    RS_XY_gpu(u0, z=500, precision="float32")
    cuda.Context.synchronize()
print("RS_XY_gpu avg ms:", (time.perf_counter()-t0)/n*1000)

