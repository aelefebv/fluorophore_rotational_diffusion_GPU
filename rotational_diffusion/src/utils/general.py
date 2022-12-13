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


def polar_displacement(x, y, z, theta_d, phi_d, method='accurate', norm=True):
    """Take a Cartesian positions x, y, z and update them by
    spherical displacements (theta_d, phi_d). Theta is how much you
    moved and phi is which way.

    Note that this returns gibberish for theta_d > pi
    """
    assert method in ('naive', 'accurate')
    x_d, y_d, z_d = to_xyz(theta_d, phi_d)
    # Since the particles aren't (generally) at the north pole, we
    # have to rotate back to each particle's actual position. We'll
    # do this via a rotation matrix calculated as described in:
    #  doi.org/10.1080/10867651.1999.10487509
    #  "Efficiently Building a Matrix to Rotate One Vector to Another",
    with np.errstate(divide='ignore'): # In case z = -1
        ovr_1pz = 1 / (1+z)
    if method == 'naive': # The obvious way
        with np.errstate(invalid='ignore'):
            x_f = x_d*(z + y*y*ovr_1pz) + y_d*(   -x*y*ovr_1pz) + z_d*(x)
            y_f = x_d*(   -x*y*ovr_1pz) + y_d*(z + x*x*ovr_1pz) + z_d*(y)
            z_f = x_d*(     -x        ) + y_d*(     -y        ) + z_d*(z)
        isnan = (z == -1) # We divided by zero above, we have to fix it now
        x_f[isnan] = -x_d[isnan]
        y_f[isnan] =  y_d[isnan]
        z_f[isnan] = -z_d[isnan]
    elif method == 'accurate': # More complicated, but numerically stable?
        # Precompute a few intermediates:
        with np.errstate(invalid='ignore'):
            y_ovr_1pz =    y*ovr_1pz #  y / (1+z)
            xy_ovr_1pz = x*y_ovr_1pz # xy / (1+z)
            yy_ovr_1pz = y*y_ovr_1pz # yy / (1+z)
            xx_ovr_1pz = x*x*ovr_1pz # xx / (1+z)
        # We divided by (1+z), which is unstable for z ~= -1
        # We'll substitute slower stable versions:
        # x^2/(1+z) = (1-z) * cos(phi)^2
        # y^2/(1+z) = (1-z) * sin(phi)^2
        # x*y/(1+z) = (1-z) * sin(phi)*cos(phi)
        unstable = z < (-1 + 5e-2) # Not sure where instability kicks in...
        x_u, y_u, z_u = x[unstable], y[unstable], z[unstable]
        phi_u = np.arctan2(y_u, x_u)
        sin_ph_u, cos_ph_u = sin_cos(phi_u)
        xy_ovr_1pz[unstable] = (1 - z_u) * sin_ph_u * cos_ph_u
        yy_ovr_1pz[unstable] = (1 - z_u) * sin_ph_u * sin_ph_u
        xx_ovr_1pz[unstable] = (1 - z_u) * cos_ph_u * cos_ph_u
        # Now we're ready for the matrix multiply:
        x_f = x_d*(z + yy_ovr_1pz) + y_d*(   -xy_ovr_1pz) + z_d*(x)
        y_f = x_d*(   -xy_ovr_1pz) + y_d*(z + xx_ovr_1pz) + z_d*(y)
        z_f = x_d*(    -x        ) + y_d*(    -y        ) + z_d*(z)
    if norm:
        r = np.sqrt(x_f*x_f + y_f*y_f + z_f*z_f)
        x_f /= r
        y_f /= r
        z_f /= r
    return x_f, y_f, z_f


def to_xyz(theta, phi, method='ugly'):
    """Convert spherical polar angles to unit-length Cartesian coordinates"""
    assert method in ('ugly', 'direct')
    sin_th, cos_th = sin_cos(theta, method='0,pi')
    sin_ph, cos_ph = sin_cos(phi, method='0,2pi')
    if method == 'direct':  # The obvious way
        x = sin_th * cos_ph
        y = sin_th * sin_ph
        z = cos_th
    elif method == 'ugly':  # An uglier method with less memory allocation
        np.multiply(sin_th, cos_ph, out=cos_ph)
        np.multiply(sin_th, sin_ph, out=sin_ph)
        x = cos_ph
        y = sin_ph
        z = cos_th
    else:
        raise ValueError("method called should either be ugly or direct.")
    return x, y, z
