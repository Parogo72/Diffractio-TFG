from functools import lru_cache


@lru_cache(maxsize=1)
def ensure_pycuda_context():
    try:
        import pycuda.driver as cuda
    except ImportError as exc:
        raise RuntimeError(
            "PyCUDA is not installed. Run: pip install pycuda"
        ) from exc

    try:
        cuda.init()
    except cuda.Error:
        pass

    try:
        current = cuda.Context.get_current()
        has_context = current is not None
    except Exception:
        has_context = False

    if not has_context:
        try:
            import pycuda.autoprimaryctx  # noqa: F401
        except Exception as exc:
            raise RuntimeError(
                "Could not create a CUDA context. "
                "Check that a CUDA-capable GPU is available."
            ) from exc

    return cuda