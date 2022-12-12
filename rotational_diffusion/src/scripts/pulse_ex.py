from rotational_diffusion import fluorophore_rotational_diffusion, pulse_schemes, fluorophores, utils
import numpy as np
import matplotlib.pyplot as plt


def plot_ratiometric_anisotropy(fluorophore, t_parallel, t_perpendicular, title,
                                x_space=(3, 6, 50), save_name=None, save=True, log=True):
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
        f'{title}\n'
        rf"$\tau_f$ = {fluorophore.state_info['singlet'].lifetime:.2f} ns, "
        rf"$\tau_d$ = {fluorophore.orientations.diffusion_time:.2f} ns"
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
    return ratiometric


def plot_counts(fluorophore, t_parallel, t_perpendicular, title,
                x_space=(-1E5, 1E6, 200), save_name=None, save=True, log=False):
    if save_name is None:
        save_name = title
    start, stop, steps = x_space
    if log:
        bins = np.logspace(start, stop, steps)
    else:
        bins = np.linspace(start, stop, steps)
    bin_centers = (bins[1:] + bins[:-1]) / 2
    (hist_parallel, _), (hist_perpendicular, _) = np.histogram(t_parallel, bins), np.histogram(t_perpendicular, bins)

    plt.figure()
    plt.plot(bin_centers, hist_parallel)
    plt.plot(bin_centers, hist_perpendicular)
    plt.suptitle(
        f'{title}\n'
        rf"$\tau_f$ = {fluorophore.state_info['singlet'].lifetime:.2f} ns, "
        rf"$\tau_d$ = {fluorophore.orientations.diffusion_time:.2f} ns"
    )
    plt.xlabel(r"Time (ns)")
    plt.ylabel('Counts')
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


my_fluorophore = fluorophores.rsEGFP2()
number_of_molecules = 1E6
my_state_info = utils.create_singlet_state_info(my_fluorophore)
f = fluorophore_rotational_diffusion.Fluorophores(
    number_of_molecules=number_of_molecules,
    state_info=my_state_info,
    diffusion_time=my_fluorophore.rotational_diffusion_time_ns
)
# pulse_schemes.starss_method1(f)
pulse_schemes.starss_method2(f)
x, y, z, t, = f.get_xyzt_at_transitions('singlet', 'ground')
xf, yf, tf = x[t > 5E05], y[t > 5E05], t[t > 5E05]
t_x, t_y = utils.split_counts_xy(xf, yf, tf)

ratiometric = plot_ratiometric_anisotropy(f, t_x, t_y, title='rsEGFP2_method2_anisotropy', save=False, x_space=(5, 6, 500))
plot_counts(f, t_x, t_y, title='rsEGFP2_method2_counts', save=False, x_space=(4E5, 7E5, 500))
