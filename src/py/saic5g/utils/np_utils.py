"""
Useful stuff numpy doesn't have
"""
import numpy as np

def interweave(*args):
    """
    Interweave as many numpy arrays as you want together.

    The arrays must be integer ratios of each other, and one
    dimensional.

    Example:
    >>> a0 = [1, 2, 3, 4]
    >>> a1 = [5, 6]
    >>> interweave(a0, a1)
    [1, 2, 5, 3, 4, 6]
    >>> interweave(a1, a0)
    [5, 1, 2, 6, 3, 4]
    """
    sizes = [a.size for a in args]
    total_size = sum(sizes)
    min_size = min(sizes)
    ns = [s/min_size for s in sizes]
    n = int(sum(ns))
    out = np.empty((total_size,), dtype=args[0].dtype)
    start = 0
    for a, an in zip(args, ns):
        ian = int(an)
        if ian != an:
            raise ValueError('Array lengths must be integer values of each other.')
        for i in range(ian):
            out[start::n] = a[i::ian]
            start += 1
    return out

def pi_wrap(angles):
    """
    Wrap angles (in radians) to (-pi, +pi]
    """
    angles = np.mod(angles, 2*np.pi)
    idx = angles > np.pi
    angles[idx] = angles[idx] - 2*np.pi
    return angles
