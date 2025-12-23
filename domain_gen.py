import numpy as np


def uniform_domain(t_begin, t_end, num_points):
    return np.linspace(t_begin, t_end, num_points)


"""
will generate n buckets 
1: base^exponent
2: base^(exponent+1)
...:
N: base^(exponent+N-1)
"""


def exponential_domain(t_begin, t_end, base, exponent, num_bukcets):
    num_points = base ** (exponent + np.arange(num_bukcets))
    diff = abs(t_end - t_begin)
    return np.vstack(
        [
            np.linspace(t_begin + diff * i, t_begin + diff * (i + 1), num_points[i])
            for i in range(num_bukcets)
        ]
    )
