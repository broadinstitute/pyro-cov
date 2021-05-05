import functools
import weakref

import torch


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


_TENSORS = {}


def deduplicate_tensor(x):
    key = x.dtype, x.stride(), x.data_ptr()
    return _TENSORS.setdefault(key, x)


def torch_map(x, **kwargs):
    """
    Calls ``leaf.to(**kwargs)`` on all tensor and module leaves of a nested
    data structure.
    """
    return _torch_map(x, **kwargs)[0]


@functools.singledispatch
def _torch_map(x):
    return x, False


@_torch_map.register(torch.Tensor)
def _(x, **kwargs):
    x_ = x.to(**kwargs)
    changed = x_ is not x
    return x_, changed


@_torch_map.register(torch.nn.Module)
def _(x, **kwargs):
    changed = True  # safe
    return x.to(**kwargs), changed


@_torch_map.register(dict)
def _(x, **kwargs):
    result = type(x)()
    changed = False
    for k, v in x.items():
        v_, changed_ = _torch_map(v)
        result[k] = v
        changed = changed or changed_
    return (result, True) if changed else (x, False)


@_torch_map.register(list)
@_torch_map.register(tuple)
def _(x, **kwargs):
    result = []
    changed = False
    for v in x:
        v_, changed_ = _torch_map(v)
        result.append(v)
        changed = changed or changed_
    result = type(x)(result)
    return (result, True) if changed else (x, False)
