from rotational_diffusion.src import components, variables, np, utils
import matplotlib.pyplot as plt

# setup
molecule_properties = components.experiment.MoleculeProperties(
    molecule=variables.molecule_properties.mScarlet,
    num_molecules=5E06,
    rotational_diffusion_time=100000 * np.pi,
)
excitation_properties = components.experiment.ExcitationProperties(
    singlet_polarization=(0, 1, 0), singlet_intensity=3,
    crescent_polarization=(1, 0, 0), crescent_intensity=0,
    trigger_polarization=(1, 0, 0), trigger_intensity=3,
    num_triggers=1,
)

# run and plot
molecule_properties.fluorophores.phototransition(
    'ground', 'singlet',
    intensity=excitation_properties.singlet_intensity,
    polarization_xyz=excitation_properties.singlet_polarization,
)
molecule_properties.fluorophores.time_evolve(10)


def save_frame(fluorophore, states=None, save=True, filepath=None):

    if states is None:
        o = fluorophore.orientations
        x, y, z = o.x, o.y, o.z
        states_vec = molecule_properties.fluorophores.states
    else:
        if not isinstance(states, list):
            states = [states]
        x, y, z, states_vec = None, None, None, None
        for state in states:
            xt, yt, zt = fluorophore.get_xyz_for_state(state)
            states_vec_temp = fluorophore.states[fluorophore.states == fluorophore.state_info[state].state_num]
            if x is None:
                x, y, z, states_vec = xt, yt, zt, states_vec_temp
            else:
                x = np.concatenate([x, xt])
                y = np.concatenate([y, yt])
                z = np.concatenate([z, zt])
                states_vec = np.concatenate([states_vec, states_vec_temp])

    fig = plt.figure(figsize=(10, 10), frameon=False)
    ax = fig.add_subplot(111, projection='3d')
    max_molecules_to_plot = 2000
    subsample = int(len(x) / max_molecules_to_plot)
    if subsample == 0:
        subsample = 1
    xs, ys, zs = x[::subsample], y[::subsample], z[::subsample]
    states_vecs = states_vec[::subsample]
    try:
        xs = xs.get()
        ys = ys.get()
        zs = zs.get()
        states_vecs = states_vecs.get()
    except TypeError:
        pass
    plot_states = states_vecs == 0  # ground
    ax.scatter(xs[plot_states], ys[plot_states], zs[plot_states], marker='.', c='black', alpha=0.025)
    plot_states = states_vecs == 1  # triplet
    ax.scatter(xs[plot_states], ys[plot_states], zs[plot_states], marker='.', c='red', alpha=0.5)
    plot_states = states_vecs == 2  # singlet
    ax.scatter(xs[plot_states], ys[plot_states], zs[plot_states], marker='.', c='green', alpha=0.5)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(25, -8)
