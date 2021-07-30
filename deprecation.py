import warnings
import functools
import inspect


def deprecated(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        frame = inspect.currentframe()
        caller = inspect.getouterframes(frame, 2)
        warnings.warn(f'{func.__name__} is deprecated but called in {caller[1][3]}')
        return func(*args, **kwargs)
    return wrapper
