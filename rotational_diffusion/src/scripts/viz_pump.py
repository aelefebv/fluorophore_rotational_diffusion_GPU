import os.path

from rotational_diffusion.src import components, variables, np, utils
import matplotlib.pyplot as plt
from datetime import datetime


def save_frame(fluorophores, ids, save=True, filepath=None, time=None):
    idxs = np.where(np.isin(fluorophores.id, ids))
    o = fluorophores.orientations
    x, y, z = o.x[idxs], o.y[idxs], o.z[idxs]
    states_vec = molecule_properties.fluorophores.states[idxs]

    fig = plt.figure(figsize=(10, 10), frameon=False)
    ax = fig.add_subplot(111, projection='3d')
    try:  # needed if on GPU
        x = x.get()
        y = y.get()
        z = z.get()
        states_vec = states_vec.get()
    except TypeError:  # gets a TypeError on CPU. There's probably a smarter way to do this, but this works for now.
        pass
    plot_states = states_vec == 0  # ground
    ax.scatter(x[plot_states], y[plot_states], z[plot_states], marker='.', c='black', alpha=0.075)
    plot_states = states_vec == 1  # triplet
    ax.scatter(x[plot_states], y[plot_states], z[plot_states], marker='.', c='red', alpha=0.33)
    plot_states = states_vec == 2  # singlet
    ax.scatter(x[plot_states], y[plot_states], z[plot_states], marker='.', c='green', alpha=0.33)

    ax.set_box_aspect((1, 1, 1))
    ax.view_init(25, -8)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_zlim(-1.1, 1.1)

    if save:
        plt.savefig(os.path.join(filepath, f'time_point-{time}ns.png'))
        plt.close()
    return None


def running_time_evolve(fluorophores, time_evolution, running_time):
    fluorophores.time_evolve(time_evolution)
    running_time += time_evolution
    return running_time


def get_ids_to_track(fluorophores, states, max_molecules):
    if states is not None:
        if not isinstance(states, list):
            states = [states]
        all_wanted_ids = np.array([])
        for state in states:
            all_wanted_ids = np.append(
                all_wanted_ids,
                fluorophores.id[fluorophores.states == fluorophores.state_info[state].state_num]
            )
    else:
        all_wanted_ids = fluorophores.id
    if len(all_wanted_ids) < max_molecules:
        return all_wanted_ids
    else:
        return all_wanted_ids[:max_molecules]


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
output_dir_top = r"C:\Users\austin\test_files\julia"

time_point = 0
dt = datetime.now().strftime('%Y%m%d%H%M%S')
output_dir = os.path.join(output_dir_top, 'output', dt)
os.makedirs(output_dir)
# animation frames:



# a. show all (i.e. only ground state) molecules diffusing
ids_to_track = get_ids_to_track(molecule_properties.fluorophores, None, 10001)
for _ in range(100):
    time_point = running_time_evolve(molecule_properties.fluorophores, 1, time_point)
    save_frame(molecule_properties.fluorophores, ids_to_track, save=True, filepath=output_dir, time=time_point)
# b. excite to singlet --> ground and singlets
# c. ground fades to 0, resulting in singlets only
# d. wait for singlets to decay --> continuous frames (maybe 0.25s resolution?) showing only singlets and triplets
#       up to delay of 25 ns
#       should see singlets disappear and triplets appear
# e. Show triplets only, diffusing until some equilibrated timepoint
#
# generate 3d projection, and 2d globe projection
# generate effective polarization ratios at each time point?

# run and plot
molecule_properties.fluorophores.phototransition(
    'ground', 'singlet',
    intensity=excitation_properties.singlet_intensity,
    polarization_xyz=excitation_properties.singlet_polarization,
)
# molecule_properties.fluorophores.delete_fluorophores_in_state('ground')
save_frame(molecule_properties.fluorophores, ids_to_track, save=True, filepath=output_dir, time=time_point)
for _ in range(100):
    time_point = running_time_evolve(molecule_properties.fluorophores, 1, time_point)
    save_frame(molecule_properties.fluorophores, ids_to_track, save=True, filepath=output_dir, time=time_point)
ids_to_track = get_ids_to_track(molecule_properties.fluorophores, 'triplet', 10001)
for _ in range(100):
    time_point = running_time_evolve(molecule_properties.fluorophores, 1, time_point)
    save_frame(molecule_properties.fluorophores, ids_to_track, save=True, filepath=output_dir, time=time_point)
# molecule_properties.fluorophores.time_evolve(10)
# save_frame(molecule_properties.fluorophores, None, save=True, filepath=output_dir)

