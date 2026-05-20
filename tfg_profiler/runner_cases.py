from importlib import import_module
from pathlib import Path
import sys


def _ensure_local_backend_available(package_name):
    """Add a local backend repository to sys.path when present in the workspace root."""
    root = Path(__file__).resolve().parents[1]
    local_repo = root / package_name
    if local_repo.exists():
        repo_path = str(local_repo)
        if repo_path not in sys.path:
            sys.path.insert(0, repo_path)


def _load_backend(package_name):
    """Load the requested diffractio-compatible backend and its required classes."""
    _ensure_local_backend_available(package_name)

    backend = import_module(package_name)
    return {
        "name": package_name,
        "np": backend.np,
        "mm": backend.mm,
        "um": backend.um,
        "Scalar_source_X": import_module(f"{package_name}.scalar_sources_X").Scalar_source_X,
        "Scalar_mask_X": import_module(f"{package_name}.scalar_masks_X").Scalar_mask_X,
        "Scalar_mask_XZ": import_module(f"{package_name}.scalar_masks_XZ").Scalar_mask_XZ,
        "Scalar_source_XY": import_module(f"{package_name}.scalar_sources_XY").Scalar_source_XY,
        "Scalar_mask_XY": import_module(f"{package_name}.scalar_masks_XY").Scalar_mask_XY,
        "Scalar_field_XYZ": import_module(f"{package_name}.scalar_fields_XYZ").Scalar_field_XYZ,
    }


def _default_wavelength(backend):
    return 0.5 * backend["um"]


def _label_suffix(package_name):
    if package_name == "diffractio":
        return ""
    return f", {package_name}"


def _setup_x_profile(size, wavelength, backend):
    np = backend["np"]
    mm = backend["mm"]
    um = backend["um"]
    x0 = np.linspace(-0.25 * mm, 0.25 * mm, size)
    u0 = backend["Scalar_source_X"](x=x0, wavelength=wavelength, info=f"u0_x_{backend['name']}")
    u0.plane_wave(A=1)
    t0 = backend["Scalar_mask_X"](x=x0, wavelength=wavelength, info=f"t0_x_{backend['name']}")
    t0.slit(x0=0, size=300 * um)
    u1 = t0 * u0
    
    if hasattr(u1, "RS_pycuda"):
        return {"func": u1.RS_pycuda, "kwargs": {"z": 1 * mm, "verbose": False}, "method_name": "RS_pycuda"}
    
    return {"func": u1.RS, "kwargs": {"z": 1 * mm, "verbose": False}}


def _setup_xz_profile(size, wavelength, backend):
    np = backend["np"]
    mm = backend["mm"]
    um = backend["um"]
    x0 = np.linspace(-100 * um, 100 * um, size)
    z_size = max(128, size // 2)
    z0 = np.linspace(100 * um, 10 * mm, z_size)
    u0 = backend["Scalar_source_X"](x=x0, wavelength=wavelength, info=f"u0_xz_{backend['name']}")
    u0.plane_wave(A=1)
    t0 = backend["Scalar_mask_X"](x=x0, wavelength=wavelength, info=f"t0_xz_{backend['name']}")
    t0.slit(x0=0, size=100 * um)
    u1 = t0 * u0
    uz = backend["Scalar_mask_XZ"](x=x0, z=z0, wavelength=wavelength, info=f"uz_xz_{backend['name']}")
    uz.incident_field(u1)
    
    if hasattr(uz, "RS_pycuda"):
        return {"func": uz.RS_pycuda, "kwargs": {"verbose": False}, "method_name": "RS_pycuda"}

    return {"func": uz.RS, "kwargs": {"verbose": False}}


def _setup_xy_profile(size, wavelength, backend):
    np = backend["np"]
    um = backend["um"]
    x0 = np.linspace(-100 * um, 100 * um, size)
    y0 = np.linspace(-100 * um, 100 * um, size)
    u0 = backend["Scalar_source_XY"](x=x0, y=y0, wavelength=wavelength, info=f"u0_xy_{backend['name']}")
    u0.plane_wave(A=1)
    t0 = backend["Scalar_mask_XY"](x=x0, y=y0, wavelength=wavelength, info=f"t0_xy_{backend['name']}")
    t0.square(r0=(0 * um, 0 * um), size=150 * um, angle=0)
    u1 = t0 * u0
    
    if hasattr(u1, "RS_pycuda"):
        return {"func": u1.RS_pycuda, "kwargs": {"z": 500 * um, "verbose": False}, "method_name": "RS_pycuda"}
    
    return {"func": u1.RS, "kwargs": {"z": 500 * um}}


def _setup_xyz_profile(size, wavelength, backend):
    np = backend["np"]
    mm = backend["mm"]
    um = backend["um"]
    x0 = np.linspace(-100 * um, 100 * um, size)
    y0 = np.linspace(-100 * um, 100 * um, size)
    z_size = max(32, size // 4)
    z0 = np.linspace(500 * um, 2 * mm, z_size)
    u0 = backend["Scalar_source_XY"](x=x0, y=y0, wavelength=wavelength, info=f"u0_xyz_{backend['name']}")
    u0.plane_wave(A=1)
    t0 = backend["Scalar_mask_XY"](x=x0, y=y0, wavelength=wavelength, info=f"t0_xyz_{backend['name']}")
    t0.square(r0=(0 * um, 0 * um), size=150 * um, angle=0)
    u1 = t0 * u0
    uz = backend["Scalar_field_XYZ"](x=x0, y=y0, z=z0, wavelength=wavelength)
    uz.incident_field(u1)

    if hasattr(uz, "RS_pycuda"):
        return {"func": uz.RS_pycuda, "kwargs": {"verbose": False}, "method_name": "RS_pycuda"}

    if hasattr(uz, "RS_batched"):
        return {"func": uz.RS_batched, "kwargs": {"batch_size": 8, "verbose": False}, "method_name": "RS_batched"}
    return {"func": uz.RS, "kwargs": {"verbose": False}, "method_name": "RS"}


def build_profile_sets(wavelength=None, package_name="diffractio"):
    """Create profile, benchmark, and sweep inputs for a diffractio-compatible backend."""
    backend = _load_backend(package_name)
    wavelength = _default_wavelength(backend) if wavelength is None else wavelength

    fixed_x = _setup_x_profile(1024 * 16, wavelength, backend)
    fixed_xz = _setup_xz_profile(1024, wavelength, backend)
    fixed_xy = _setup_xy_profile(1024, wavelength, backend)
    fixed_xyz = _setup_xyz_profile(256, wavelength, backend)

    suffix = _label_suffix(package_name)
    xyz_prefix = fixed_xyz.get("method_name", "RS")

    fixed_x_profile = {"func": fixed_x["func"], "kwargs": dict(fixed_x["kwargs"])}
    fixed_xz_profile = {"func": fixed_xz["func"], "kwargs": dict(fixed_xz["kwargs"])}
    fixed_xy_profile = {"func": fixed_xy["func"], "kwargs": dict(fixed_xy["kwargs"])}
    fixed_xyz_profile = {"func": fixed_xyz["func"], "kwargs": dict(fixed_xyz["kwargs"])}

    if "verbose" in fixed_x_profile["kwargs"]:
        fixed_x_profile["kwargs"]["verbose"] = True
    if "verbose" in fixed_xz_profile["kwargs"]:
        fixed_xz_profile["kwargs"]["verbose"] = True
    if "verbose" in fixed_xyz_profile["kwargs"]:
        fixed_xyz_profile["kwargs"]["verbose"] = True

    profiles = [
        {"label": f"RS X (Scalar_field_X{suffix})", "func": fixed_x_profile["func"], "kwargs": fixed_x_profile["kwargs"]},
        {"label": f"RS XZ (Scalar_mask_XZ{suffix})", "func": fixed_xz_profile["func"], "kwargs": fixed_xz_profile["kwargs"]},
        {"label": f"RS XY (Scalar_field_XY{suffix})", "func": fixed_xy_profile["func"], "kwargs": fixed_xy_profile["kwargs"]},
        {"label": f"{xyz_prefix} XYZ (Scalar_field_XYZ{suffix})", "func": fixed_xyz_profile["func"], "kwargs": fixed_xyz_profile["kwargs"]},
    ]

    benchmark_profiles = [
        {"label": f"RS X (Scalar_field_X{suffix})", "func": fixed_x["func"], "kwargs": fixed_x["kwargs"]},
        {"label": f"RS XZ (Scalar_mask_XZ{suffix})", "func": fixed_xz["func"], "kwargs": fixed_xz["kwargs"]},
        {"label": f"RS XY (Scalar_field_XY{suffix})", "func": fixed_xy["func"], "kwargs": fixed_xy["kwargs"]},
        {"label": f"{xyz_prefix} XYZ (Scalar_field_XYZ{suffix})", "func": fixed_xyz["func"], "kwargs": fixed_xyz["kwargs"]},
    ]

    sweep_profiles = [
        {"label": f"RS X (Scalar_field_X{suffix})", "setup": lambda size: _setup_x_profile(size, wavelength, backend)},
        {"label": f"RS XZ (Scalar_mask_XZ{suffix})", "setup": lambda size: _setup_xz_profile(size, wavelength, backend)},
        {"label": f"RS XY (Scalar_field_XY{suffix})", "setup": lambda size: _setup_xy_profile(size, wavelength, backend)},
        {"label": f"{xyz_prefix} XYZ (Scalar_field_XYZ{suffix})", "setup": lambda size: _setup_xyz_profile(size, wavelength, backend)},
    ]

    return profiles, benchmark_profiles, sweep_profiles
