from rotational_diffusion import fluorophore_rotational_diffusion
from numpy.random import uniform
import matplotlib.pyplot as plt
import numpy as np


def create_triplet_state_info(fluorophore):
    state_info = fluorophore_rotational_diffusion.FluorophoreStateInfo()
    state_info.add('ground')
    state_info.add('triplet', lifetime=fluorophore.triplet_lifetime_ns, final_states='ground')
    # state_info.add('singlet', lifetime=fluorophore.singlet_lifetime_ns, final_states='ground')
    state_info.add(
        'singlet', lifetime=fluorophore.singlet_lifetime_ns,
        final_states=['ground', 'triplet'],
        probabilities=[1-fluorophore.triplet_quantum_yield, fluorophore.triplet_quantum_yield]
    )
    return state_info


def create_singlet_state_info(fluorophore):
    state_info = fluorophore_rotational_diffusion.FluorophoreStateInfo()
    state_info.add('ground')
    state_info.add(
        'singlet', lifetime=fluorophore.singlet_lifetime_ns,
        final_states='ground',
    )
    return state_info


def split_counts_xy(x, y, t):
    p_x, p_y = x**2, y**2
    r = uniform(0, 1, size=len(t))
    in_channel_x = (r < p_x)
    in_channel_y = (p_x <= r) & (r < p_x + p_y)
    t_x, t_y = t[in_channel_x], t[in_channel_y]
    return t_x, t_y


def plot_ratiometric_anisotropy(fluorophore, t_parallel, t_perpendicular, title,
                                x_space=(3, 6, 50), save_name=None, save=True, log=False):
    if save_name is None:
        save_name = title
    start, stop, steps = x_space
    if log:
        bins = np.logspace(start, stop, steps)
    else:
        bins = np.linspace(start, stop, steps)
    bin_centers = (bins[1:] + bins[:-1]) / 2
    (hist_parallel, _), (hist_perpendicular, _) = np.histogram(t_parallel, bins), np.histogram(t_perpendicular, bins)

    ratiometric = (hist_parallel - hist_perpendicular) / (hist_parallel + 2 * hist_perpendicular)

    plt.figure()
    plt.plot(bin_centers, ratiometric, '.-')
    plt.suptitle(
        f'{title}'
        rf"$\tau_f$ = {fluorophore.state_info['singlet'].lifetime} ns, "
        rf"$\tau_d$ = {fluorophore.orientations.diffusion_time} ns"
    )
    plt.xlabel(r"Time (ns)")
    plt.ylabel('Anisotropy')
    if log:
        plt.xscale('log')
    # plt.legend()
    plt.grid('on')
    if save:
        plt.savefig(save_name)
        plt.close()
    else:
        plt.show()
    return None
