import time
from functools import wraps


def timed(name: str | None = None):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()

            try:
                return func(*args, **kwargs)
            finally:
                elapsed = time.perf_counter() - start
                label = name or func.__name__
                print(f"[TIMING] {label}: {elapsed:.3f}s")

        return wrapper

    return decorator
