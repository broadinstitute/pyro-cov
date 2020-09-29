import functools
import weakref


def weak_memoize_by_id(fn):
    cache = {}
    missing = object()  # An arbitrary value that cannot be returned by fn.

    @functools.wraps(fn)
    def memoized_fn(*args):
        key = tuple(map(id, args))
        result = cache.get(key, missing)
        if result is missing:
            result = cache[key] = fn(*args)
            for arg in args:
                # Register callbacks only for types that support weakref.
                if type(arg).__weakrefoffset__:
                    weakref.finalize(arg, cache.pop, key, None)
        return result

    return memoized_fn
