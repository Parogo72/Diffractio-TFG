import numpy as np
from concurrent.futures import ProcessPoolExecutor

from tfg_profiler.runner_cases import build_profile_sets


CASES = ("X", "XZ", "XY", "XYZ")

def _case_name_from_label(label):
    upper = label.upper()
    if " XYZ " in f" {upper} ":
        return "XYZ"
    if " XZ " in f" {upper} ":
        return "XZ"
    if " XY " in f" {upper} ":
        return "XY"
    if " X " in f" {upper} ":
        return "X"
    raise ValueError(f"Could not infer case from label: {label}")


def _profiles_by_case(package_name):
    _, benchmark_profiles, _ = build_profile_sets(package_name=package_name)
    cases = {}
    for item in benchmark_profiles:
        case = _case_name_from_label(item["label"])
        cases[case] = item

    missing = [case for case in CASES if case not in cases]
    if missing:
        raise ValueError(f"Missing profile cases for {package_name}: {missing}")

    return cases


def _run_profile_item(item):
    func = item["func"]
    args = item.get("args", ())
    kwargs = item.get("kwargs", {})
    func(*args, **kwargs)
    owner = getattr(func, "__self__", None)

    if owner is None:
        raise TypeError("Profile callable is expected to be a bound method.")

    return owner


def _run_profile_case_isolated(package_name, case):
    """Run one profile case in a fresh process to avoid backend state leakage."""
    cases = _profiles_by_case(package_name)
    output = _run_profile_item(cases[case])
    return _extract_numeric_array(output)


def _extract_numeric_array(field_owner):
    for attr in ("u", "field", "n"):
        value = getattr(field_owner, attr, None)
        if value is None:
            continue

        if hasattr(value, "get") and callable(value.get):
            value = value.get()

        arr = np.asarray(value)
        if arr.dtype != object:
            return arr

    raise TypeError(f"Unable to extract numeric array from type: {type(field_owner)!r}")


def _compare_arrays(array_a, array_b, rtol, atol):
    same_shape = array_a.shape == array_b.shape
    exact_equal = bool(same_shape and np.array_equal(array_a, array_b, equal_nan=True))
    allclose = bool(same_shape and np.allclose(array_a, array_b, rtol=rtol, atol=atol, equal_nan=True))
    if not same_shape:
        return {"same_shape": False, "exact_equal": False, "allclose": False, "max_abs_diff": float("inf")}

    if array_a.size == 0:
        max_abs_diff = 0.0
    else:
        max_abs_diff = float(np.max(np.abs(array_a - array_b)))

    return {"same_shape": True, "exact_equal": exact_equal, "allclose": allclose, "max_abs_diff": max_abs_diff}


def run_profile_output_check(package_a, package_b, *, rtol=1e-4, atol=1e-6):
    """Compare one execution output for X, XZ, XY, and XYZ between two packages."""
    print(f"\nOutput comparison for {package_a} vs {package_b}")
    print(f"Cases: {', '.join(CASES)}")
    print(f"Tolerance: rtol={rtol}, atol={atol}\n")

    results = {}
    for case in CASES:
        with ProcessPoolExecutor(max_workers=1) as executor:
            array_a = executor.submit(_run_profile_case_isolated, package_a, case).result()

        with ProcessPoolExecutor(max_workers=1) as executor:
            array_b = executor.submit(_run_profile_case_isolated, package_b, case).result()

        metrics = _compare_arrays(array_a, array_b, rtol=rtol, atol=atol)
        results[case] = metrics

        status = "OK" if metrics["allclose"] else "DIFF"
        print(
            f"[{case}] {status} | shape_a={array_a.shape} shape_b={array_b.shape} "
            f"exact={metrics['exact_equal']} max_abs={metrics['max_abs_diff']:.6e}"
        )

    exact_count = sum(1 for metrics in results.values() if metrics["exact_equal"])
    close_count = sum(1 for metrics in results.values() if metrics["allclose"])
    print(f"\nSummary: exact={exact_count}/{len(CASES)}, allclose={close_count}/{len(CASES)}")
    return results
