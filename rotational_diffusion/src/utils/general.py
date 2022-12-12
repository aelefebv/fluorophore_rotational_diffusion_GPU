import numpy as np
from numpy.random import uniform


def split_counts_xy(x, y, t):
    p_x, p_y = x**2, y**2
    r = uniform(0, 1, size=len(t))
    in_channel_x = (r < p_x)
    in_channel_y = (p_x <= r) & (r < p_x + p_y)
    t_x, t_y = t[in_channel_x], t[in_channel_y]
    return t_x, t_y


def sin_cos(radians, method='sqrt'):
    """We often want both the sine and cosine of an array of angles. We
    can do this slightly faster with a sqrt, especially in the common
    cases where the angles are between 0 and pi, or 0 and 2pi.

    Since the whole point of this code is to be fast, there's no
    checking for validity, i.e. 0 < radians < pi, 2pi. Make sure you
    don't use out-of-range arguments.
    """
    radians = np.atleast_1d(radians)
    assert method in ('direct', 'sqrt', '0,2pi', '0,pi')
    cos = np.cos(radians)

    if method == 'direct':  # Simple and obvious
        sin = np.sin(radians)
    else:  # |sin| = np.sqrt(1 - cos*cos)
        sin = np.sqrt(1 - cos*cos)

    if method == 'sqrt': # Handle arbitrary values of 'radians'
        sin[np.pi - (radians % (2*np.pi)) < 0] *= -1
    elif method == '0,2pi': # Assume 0 < radians < 2pi, no mod
        sin[np.pi - (radians            ) < 0] *= -1
    elif method == '0,pi': # Assume 0 < radians < pi, no negation
        pass
    return sin, cos
