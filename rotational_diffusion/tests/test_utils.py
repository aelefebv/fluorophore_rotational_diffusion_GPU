import numpy as np
from numpy.random import uniform

from rotational_diffusion.src.utils.general import sin_cos


def _test_sin_cos(n=int(1e6), dtype='float64'):
    import time
    """This test doesn't pass for float32, but neither does sin^2 + cos^2 == 1
    """
    theta = uniform(0,   np.pi, size=n).astype(dtype)
    phi   = uniform(0, 2*np.pi, size=n).astype(dtype)
    x     = uniform(0, 8*np.pi, size=n).astype(dtype)

    assert np.allclose(sin_cos(x, method='direct'),
                       sin_cos(x, method='sqrt'))

    assert np.allclose(sin_cos(phi, method='direct'),
                       sin_cos(phi, method='0,2pi'))

    assert np.allclose(sin_cos(theta, method='direct'),
                       sin_cos(theta, method='0,pi'))
    print()
    t = {}
    for method in ('direct', 'sqrt', '0,2pi', '0,pi'):
        t[method] = []
        for i in range(10):
            start = time.perf_counter()
            sin_cos(theta, method)
            end = time.perf_counter()
            t[method].append(end - start)
        print('%0.1f'%(1e9*np.mean(t[method]) / n),
              'nanoseconds per %6s sin_cos()'%(method))
    return None
