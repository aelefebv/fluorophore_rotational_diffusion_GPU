import numpy as np
from matplotlib import pyplot as plt


def plot_ratiometric_anisotropy(fluorophore, t_parallel, t_perpendicular, title,
                                x_space=(3, 6, 50), save_name=None, save=True, log=True, plot_keep=None):
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
    if plot_keep is None:
        plot_keep = plt.gca()
    plot_keep.plot(bin_centers, ratiometric, '.-')
    plot_keep.set(xlabel="Time (ns)", ylabel='Anisotropy',
                  title=f'{title}\n')
                        # rf"$\tau_f$ = {fluorophore.state_info['singlet'].lifetime:.2f} ns,"
                        # rf"$\tau_d$ = {fluorophore.orientations.rot_diffusion_time:.2f} ns")
    # plot_keep.suptitle(
    #     f'{title}\n'
    #     rf"$\tau_f$ = {fluorophore.state_info['singlet'].lifetime:.2f} ns, "
    #     rf"$\tau_d$ = {fluorophore.orientations.rot_diffusion_time:.2f} ns"
    # )
    # plot_keep.xlabel(r"Time (ns)")
    # plot_keep.ylabel('Anisotropy')
    if log:
        plot_keep.xscale('log')
    # plt.legend()
    plot_keep.grid('on')
    # if save:
    #     plot_keep.savefig(save_name)
    #     plot_keep.close()
    # else:
    #     plot_keep.show()
    return plot_keep


def plot_counts(fluorophore, t_parallel, t_perpendicular, title,
                x_space=(-1E5, 1E6, 200), save_name=None, save=True, log=False, plot_keep=None):
    if save_name is None:
        save_name = title
    start, stop, steps = x_space
    if log:
        bins = np.logspace(start, stop, steps)
    else:
        bins = np.linspace(start, stop, steps)
    bin_centers = (bins[1:] + bins[:-1]) / 2
    (hist_parallel, _), (hist_perpendicular, _) = np.histogram(t_parallel, bins), np.histogram(t_perpendicular, bins)

    if plot_keep is None:
        plot_keep = plt.gca()
    plot_keep.plot(bin_centers, hist_parallel)
    plot_keep.plot(bin_centers, hist_perpendicular)
    plot_keep.set(xlabel=r"Time (ns)", ylabel='Counts',
                  title=f'{title}\n')
                        # rf"$\tau_f$ = {fluorophore.state_info['singlet'].lifetime:.2f} ns,"
                        # rf"$\tau_d$ = {fluorophore.orientations.rot_diffusion_time:.2f} ns")
    # plot_keep.xlabel(r"Time (ns)")
    # plot_keep.ylabel('Counts')
    if log:
        plot_keep.xscale('log')
    # plt.legend()
    plot_keep.grid('on')
    # if save:
    #     plot_keep.savefig(save_name)
    #     plot_keep.close()
    # else:
    #     plot_keep.show()
    return plot_keep
