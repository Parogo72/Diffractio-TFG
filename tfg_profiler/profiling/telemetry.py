import os

try:
    import psutil
except Exception:
    psutil = None


def get_rss_bytes():
    """Return process RSS in bytes when psutil is available."""
    if psutil is None:
        return None
    return psutil.Process(os.getpid()).memory_info().rss
