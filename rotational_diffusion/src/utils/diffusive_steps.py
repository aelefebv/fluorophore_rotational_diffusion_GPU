from rotational_diffusion.src.utils import general
from rotational_diffusion.src import np


def ghosh_propagator(step_sizes):
    """
    Draw random angular displacements from Ghosh propagator for diffusion on a sphere.

    The propagator, as described in arXiv:1303.1278, provides a more accurate estimation
    for larger time steps compared to a Gaussian propagator. This function uses an iterative
    approach to draw random numbers until all step sizes have been drawn.

    Parameters:
    step_sizes (np.ndarray): 1D array of nonnegative floats representing 'sigma' in the equation

    Returns:
    np.ndarray: 1D array with same shape as 'step_sizes', where each entry is a random
                number drawn from a distribution determined by the corresponding entry of 'step_sizes'.
    """

    min_step_sizes = np.min(step_sizes)
    assert min_step_sizes >= 0
    if min_step_sizes == 0:
        step_sizes = np.clip(step_sizes, a_min=1e-12, a_max=None)
    # Iteratively populate the result vector:
    first_iteration = True
    while True:
        # Which step sizes do we still need to draw random numbers for?
        steps = step_sizes if first_iteration else step_sizes[tbd]
        # Draw from a truncated non-normalized version of the Gaussian
        # propagator as an upper bound for rejection sampling. Don't
        # bother drawing values that will exceed pi.
        will_draw_pi = np.exp(-(np.pi / steps)**2)
        candidates = steps * np.sqrt(-np.log(
            np.random.uniform(will_draw_pi, 1, len(steps))))
        # To convert draws from our upper bound distribution to our desired
        # distribution, reject samples stochastically by the ratio of the
        # desired distribution to the upper bound distribution,
        # which is sqrt(sin(x)/x).
        rejected = (np.random.uniform(0, 1, candidates.shape) >
                    np.sqrt(np.sin(candidates) / candidates))
        # Update results
        if first_iteration:
            result = candidates
            tbd = np.nonzero(rejected)[0]  # Coordinates of unset results
            first_iteration = False
        else:
            result[tbd] = candidates
            tbd = tbd[rejected]
        if len(tbd) == 0:
            break  # We've set every element of the result
    return result


def gaussian_propagator(step_sizes):
    """
    Draw random angular displacements from Gaussian propagator for diffusion on a sphere.

    The Gaussian propagator, described in arXiv:1303.1278, is accurate for small time steps. This function
    uses inverse transform sampling to draw random numbers from the Gaussian distribution.

    Parameters:
    step_sizes (np.ndarray): 1D array of nonnegative floats representing 'sigma' in the equation

    Returns:
    np.ndarray: 1D array with same shape as 'step_sizes', where each entry is a random
                number drawn from a distribution determined by the corresponding entry of 'step_sizes'.
    """

    # Calculate draws via inverse transform sampling.
    result = step_sizes * np.sqrt(-np.log(np.random.uniform(0, 1, len(step_sizes))))
    return result


def diffusive_step(x, y, z, normalized_time_step, propagator='ghosh'):
    """
    Perform a diffusive step on the sphere.

    Parameters:
    x, y, z (np.ndarray): 1D arrays representing 3D Cartesian coordinates
    normalized_time_step (float): Normalized time step value
    propagator (str): Type of propagator to use ('ghosh' or 'gaussian')

    Returns:
    tuple: x, y, z after diffusive step
    """
    assert len(x) == len(y) == len(z)
    angle_step = np.sqrt(2*normalized_time_step)
    assert angle_step.shape in ((), (1,), x.shape)
    angle_step = np.broadcast_to(angle_step, x.shape)
    assert propagator in ('ghosh', 'gaussian')
    prop = ghosh_propagator if propagator == 'ghosh' else gaussian_propagator
    theta_d = prop(angle_step)
    phi_d = np.random.uniform(0, 2*np.pi, len(angle_step))
    return general.polar_displacement(x, y, z, theta_d, phi_d)


def safe_diffusive_step(
    x, y, z,
    normalized_time_step,
    max_safe_step=0.5,  # Don't count on this, could be wrong
):
    """
    Perform a diffusive step on the sphere with a 'safe' maximum step size.

    Parameters:
    x, y, z (np.ndarray): 1D arrays representing 3D Cartesian coordinates
    normalized_time_step (float): Normalized time step value
    max_safe_step (float): Maximum safe step size. Default is 0.5, but this could be inaccurate.

    Returns:
    tuple: x, y, z after safe diffusive step
    """

    num_steps, remainder = np.divmod(normalized_time_step, max_safe_step)
    num_steps = num_steps.astype('uint64')  # Always an integer
    num_steps_min = np.amin(num_steps)
    num_steps_max = np.amax(num_steps)
    if num_steps_min == num_steps_max:  # Scalar time step
        for _ in range(int(num_steps_max)):
            x, y, z = diffusive_step(x, y, z, max_safe_step)
    else:  # Vector time step
        assert len(normalized_time_step) == len(x)
        t_is_sorted = np.all(np.diff(normalized_time_step) >= 0)
        if not t_is_sorted:  # Sorted xyz makes selecting unfinished stuff fast
            idx = np.argsort(num_steps)
            x, y, z = x[idx], y[idx], z[idx]
            num_steps = num_steps[idx]
        which_step = 1
        while True:
            first_unfinished = np.searchsorted(num_steps, np.array(which_step))
            if first_unfinished == len(num_steps): # We're done taking steps
                break
            s = slice(first_unfinished, None)
            x[s], y[s], z[s] = diffusive_step(x[s], y[s], z[s], max_safe_step)
            which_step += 1
    # Finally, take our 'remainder' step:
    if np.amax(remainder) > 0:
        x, y, z = diffusive_step(x, y, z, remainder)
    return x, y, z
