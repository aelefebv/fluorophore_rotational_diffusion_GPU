import os.path

from rotational_diffusion.src import components, variables, np, utils
import matplotlib.pyplot as plt
from datetime import datetime
import numpy


plt.ioff()

def convert_3d_to_theta_phi(x, y, z):
    theta = numpy.arccos(z)
    phi = numpy.arctan2(y, x)
    return numpy.degrees(theta), numpy.degrees(phi) + 180


def save_frame(fluorophores, ids,
               save=True, filepath=None,
               frame_num=None,
               projection_type='3d',
               view_angle=(25, -8),
               subdir_name='3d'):

    alpha_0 = 0.01
    alpha_1 = 0.5
    alpha_2 = 0.25

    idxs = np.where(np.isin(fluorophores.id, ids))
    o = fluorophores.orientations
    x, y, z = o.x[idxs], o.y[idxs], o.z[idxs]
    states_vec = molecule_properties.fluorophores.states[idxs]

    try:  # needed if on GPU
        x = x.get()
        y = y.get()
        z = z.get()
        states_vec = states_vec.get()
    except TypeError:  # gets a TypeError on CPU. There's probably a smarter way to do this, but this works for now.
        pass

    fig = plt.figure(figsize=(10, 10), frameon=False)
    if projection_type == '3d':
        ax = fig.add_subplot(111, projection='3d')
        plot_states = states_vec == 0  # ground
        ax.scatter(x[plot_states], y[plot_states], z[plot_states], marker='.', c='black', alpha=alpha_0)
        plot_states = states_vec == 2  # singlet
        ax.scatter(x[plot_states], y[plot_states], z[plot_states], marker='.', c='green', alpha=alpha_2)
        plot_states = states_vec == 1  # triplet
        ax.scatter(x[plot_states], y[plot_states], z[plot_states], marker='.', c='red', alpha=alpha_1)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(*view_angle)
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.set_zlim(-1.1, 1.1)
    elif projection_type == 'aitoff':
        ax = fig.add_subplot(111, projection='aitoff')
        plot_states = states_vec == 0  # ground
        theta, phi = convert_3d_to_theta_phi(x[plot_states], y[plot_states], z[plot_states])
        ax.scatter(theta, phi, marker='.', c='black', alpha=alpha_0)
        plot_states = states_vec == 2  # singlet
        theta, phi = convert_3d_to_theta_phi(x[plot_states], y[plot_states], z[plot_states])
        ax.scatter(theta, phi, marker='.', c='green', alpha=alpha_2)
        plot_states = states_vec == 1  # triplet
        theta, phi = convert_3d_to_theta_phi(x[plot_states], y[plot_states], z[plot_states])
        ax.scatter(theta, phi,  marker='.', c='red', alpha=alpha_1)
        plt.grid(True)
    if save:
        subdir = os.path.join(filepath, subdir_name)
        os.makedirs(subdir, exist_ok=True)
        plt.savefig(os.path.join(subdir, f'{frame_num:09}.png'))
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


def write_gif_from_folder(png_dir, gif_output_dir, str_template='%09d.png', file_name='output.gif', fps=10, width=1000):
    output_path = os.path.join(gif_output_dir, file_name)
    input_path = os.path.join(png_dir, str_template)
    palette_path = os.path.join(gif_output_dir, "palette.png")

    palette_gen = f'ffmpeg -y -i {input_path} -vf fps={fps},scale={width}:-1:flags=lanczos,palettegen {palette_path}'
    gif_gen = f'ffmpeg -y -i {input_path} -i {palette_path} -filter_complex fps={fps},' \
              f'scale={width}:-1:flags=lanczos[x];[x][1:v]paletteuse {output_path}'

    os.system(palette_gen)
    os.system(gif_gen)
    os.remove(palette_path)
    return None


# setup
output_dir_top = r"C:\Users\austin\test_files\julia"
# num_frames = 10
frame_num = 0
num_molecules = 1E6
time_step_ns = 0.5

time_point = 0
dt = datetime.now().strftime('%Y%m%d%H%M%S')
output_dir = os.path.join(output_dir_top, 'output', dt)
os.makedirs(output_dir)

molecule_properties = components.experiment.MoleculeProperties(
    molecule=variables.molecule_properties.mScarlet,
    num_molecules=num_molecules,
    rotational_diffusion_time=5000 * np.pi,
)
excitation_properties = components.experiment.ExcitationProperties(
    singlet_polarization=(0, 1, 0), singlet_intensity=3,
    crescent_polarization=(1, 0, 0), crescent_intensity=0,
    trigger_polarization=(1, 0, 0), trigger_intensity=3,
    num_triggers=1,
)

# animation frames:
# a. show all (i.e. only ground state) molecules diffusing
ids_to_track = get_ids_to_track(molecule_properties.fluorophores, None, num_molecules)
for _ in range(10):
    frame_num += 1
    time_point = running_time_evolve(molecule_properties.fluorophores, time_step_ns, time_point)
    save_frame(molecule_properties.fluorophores, ids_to_track,
               save=True, filepath=output_dir, frame_num=frame_num)
    save_frame(molecule_properties.fluorophores, ids_to_track,
               save=True, filepath=output_dir, frame_num=frame_num, view_angle=(0, 0), subdir_name='2d_proj')
# generate effective polarization ratios at each time point?
# b. excite to singlet --> ground and singlets
molecule_properties.fluorophores.phototransition(
    'ground', 'singlet',
    intensity=excitation_properties.singlet_intensity,
    polarization_xyz=excitation_properties.singlet_polarization,
)
molecule_properties.fluorophores.delete_fluorophores_in_state('ground')
frame_num += 1
save_frame(molecule_properties.fluorophores, ids_to_track,
           save=True, filepath=output_dir, frame_num=frame_num)
save_frame(molecule_properties.fluorophores, ids_to_track,
           save=True, filepath=output_dir, frame_num=frame_num, view_angle=(0, 0), subdir_name='2d_proj')
# c. wait for singlets to decay --> continuous frames (maybe 0.25s resolution?) showing only singlets and triplets
#       up to delay of 25 ns
#       should see singlets disappear and triplets appear
for _ in range(50):
    frame_num += 1
    time_point = running_time_evolve(molecule_properties.fluorophores, time_step_ns, time_point)
    molecule_properties.fluorophores.delete_fluorophores_in_state('ground')
    save_frame(molecule_properties.fluorophores, ids_to_track,
               save=True, filepath=output_dir, frame_num=frame_num)
    save_frame(molecule_properties.fluorophores, ids_to_track,
               save=True, filepath=output_dir, frame_num=frame_num, view_angle=(0, 0), subdir_name='2d_proj')
# e. Show triplets only, diffusing until some equilibrated timepoint
molecule_properties.fluorophores.delete_fluorophores_in_state('ground')
ids_to_track = get_ids_to_track(molecule_properties.fluorophores, 'triplet', num_molecules)
triplet_times = numpy.logspace(-2, 1, num=100)
for _, triplet_time in enumerate(triplet_times):
    frame_num += 1
    time_point = running_time_evolve(molecule_properties.fluorophores, triplet_time, time_point)
    save_frame(molecule_properties.fluorophores, ids_to_track,
               save=True, filepath=output_dir, frame_num=frame_num)
    save_frame(molecule_properties.fluorophores, ids_to_track,
               save=True, filepath=output_dir, frame_num=frame_num, view_angle=(0, 0), subdir_name='2d_proj')
for _ in range(1000):
    frame_num += 1
    time_point = running_time_evolve(molecule_properties.fluorophores, triplet_times[-1], time_point)
    save_frame(molecule_properties.fluorophores, ids_to_track,
               save=True, filepath=output_dir, frame_num=frame_num)
    save_frame(molecule_properties.fluorophores, ids_to_track,
               save=True, filepath=output_dir, frame_num=frame_num, view_angle=(0, 0), subdir_name='2d_proj')

subdirs = ['3d', '2d_proj']
for subdir in subdirs:
    subdir_path = os.path.join(output_dir, subdir)
    write_gif_from_folder(subdir_path, output_dir, file_name=f'{subdir}.gif')


