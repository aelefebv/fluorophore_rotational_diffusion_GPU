from rotational_diffusion.src import components, variables, np, utils

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
molecule_properties.fluorophores.time_evolve(10.5)

import matplotlib.pyplot as plt
save = False
state = 'triplet'  # triplet, singlet, ground, None (for all)
fluorophore = molecule_properties.fluorophores

if state is None:
    o = fluorophore.orientations
    x, y, z = o.x, o.y, o.z
    states = molecule_properties.fluorophores.states
else:
    x, y, z = fluorophore.get_xyz_for_state(state)
    states = fluorophore.states[fluorophore.states==fluorophore.state_info[state].state_num]

fig = plt.figure(figsize=(10, 10), frameon=False)
ax = fig.add_subplot(111, projection='3d')
max_molecules_to_plot = 2000
subsample = int(len(x) / max_molecules_to_plot)
if subsample == 0:
    subsample = 1
xs, ys, zs = x[::subsample], y[::subsample], z[::subsample]
statess = states[::subsample]
try:
    xs = xs.get()
    ys = ys.get()
    zs = zs.get()
    statess = statess.get()
except TypeError:
    pass
plot_states = statess == 0  # ground
ax.scatter(xs[plot_states], ys[plot_states], zs[plot_states], marker='.', c='black', alpha=0.025)
plot_states = statess == 1  # triplet
ax.scatter(xs[plot_states], ys[plot_states], zs[plot_states], marker='.', c='red')
plot_states = statess == 2  # singlet
ax.scatter(xs[plot_states], ys[plot_states], zs[plot_states], marker='.', c='green')
ax.set_box_aspect((1, 1, 1))
ax.view_init(25, -8)
