def _apply(x, f):

    if isinstance(x, (list, tuple)):
        if len(x) and sum([isinstance(xi, (str, int, float)) for xi in x]) == len(x):
            return _print(f"[{repr(x[0])}, ...]")
        return x.__class__([_apply(xi, f) for xi in x])

    if not isinstance(x, dict):
        if _has_method(x, "to_dict"):
            x = x.to_dict()
        if _has_method(x, "as_dict"):
            x = x.as_dict()
        if _has_method(x, "items"):
            x = dict(x)

    if isinstance(x, dict):
        return {k: _apply(v, f) for k, v in x.items()}

    try:
        return f(x)
    except TypeError:
        return _print(f"<Unknown Type: {x.__class__.__name__}>")
    except Exception as e:
        return _print(f"<{e}>")

def desc(x):
    from torch import Tensor
    from rich.pretty import pprint

    def _desc(_x):
        if isinstance(_x, (Tensor,)):
            dtype = str(_x.dtype).replace("torch.", "")
            return _print(f"Tensor({tuple(_x.shape)}, {dtype})")
        if isinstance(_x, (int, float, str)):
            return _print(_x)
        raise TypeError

    pprint(_apply(x, _desc))


class _print(str):
    def __repr__(self) -> str:
        return self


def _has_method(obj, methodname) -> bool:
    return getattr(getattr(obj, methodname, None), "__call__", False) is not False
