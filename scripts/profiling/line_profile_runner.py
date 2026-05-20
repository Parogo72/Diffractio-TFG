from diffractio import mm, um, np
from diffractio.scalar_sources_XY import Scalar_source_XY
from diffractio.scalar_masks_XY import Scalar_mask_XY


try:
    profile
except NameError:
    def profile(func):
        return func


def build_xy_case(size):
    wavelength = 0.5 * um
    x0 = np.linspace(-100 * um, 100 * um, size)
    y0 = np.linspace(-100 * um, 100 * um, size)
    u0 = Scalar_source_XY(x=x0, y=y0, wavelength=wavelength, info="u0_xy")
    u0.plane_wave(A=1)
    t0 = Scalar_mask_XY(x=x0, y=y0, wavelength=wavelength, info="t0_xy")
    t0.square(r0=(0 * um, 0 * um), size=150 * um, angle=0)
    u1 = t0 * u0
    return u1


@profile
def profile_kernel_rs(size=1024, z=500 * um):
    u1 = build_xy_case(size)

    return u1.RS(z=z)


if __name__ == "__main__":

    profile_kernel_rs()
