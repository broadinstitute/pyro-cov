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
def _torch_map(x, **kwargs):
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
        v, v_changed = _torch_map(v, **kwargs)
        result[k] = v
        changed = changed or v_changed
    return (result, True) if changed else (x, False)


@_torch_map.register(list)
@_torch_map.register(tuple)
def _(x, **kwargs):
    result = []
    changed = False
    for v in x:
        v, v_changed = _torch_map(v, **kwargs)
        result.append(v)
        changed = changed or v_changed
    result = type(x)(result)
    return (result, True) if changed else (x, False)


def pretty_print(x, *, name="", max_items=10):
    if isinstance(x, (int, float, str, bool)):
        print(f"{name} = {repr(x)}")
    elif isinstance(x, torch.Tensor):
        print(f"{name}: {type(x).__name__} of shape {tuple(x.shape)}")
    elif isinstance(x, (tuple, list)):
        print(f"{name}: {type(x).__name__} of length {len(x)}")
    elif isinstance(x, dict):
        print(f"{name}: {type(x).__name__} of length {len(x)}")
        if len(x) <= max_items:
            for k, v in x.items():
                pretty_print(v, name=f"{name}[{repr(k)}]", max_items=max_items)
    else:
        print(f"{name}: {type(x).__name__}")
