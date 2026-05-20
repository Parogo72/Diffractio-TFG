from line_profiler import LineProfiler

from diffractio import um, np
from diffractio.scalar_sources_XY import Scalar_source_XY
from diffractio.scalar_masks_XY import Scalar_mask_XY
from diffractio.scalar_fields_XY import Scalar_field_XY


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


def run_rs(size=1024, z=500 * um):
    u1 = build_xy_case(size)
    return u1.RS(z=z)


def _unwrap_check_none(func):
    if func is None or func.__closure__ is None:
        return func
    for cell in func.__closure__:
        try:
            candidate = cell.cell_contents
        except ValueError:
            continue
        if callable(candidate):
            return candidate
    return func


def main():
    profiler = LineProfiler()

    if hasattr(Scalar_field_XY, "_RS_"):
        profiler.add_function(_unwrap_check_none(Scalar_field_XY._RS_))

    profiled = profiler(run_rs)
    profiled()
    profiler.print_stats()


if __name__ == "__main__":
    main()
