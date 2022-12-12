import numpy as np
from matplotlib import pyplot as plt


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
