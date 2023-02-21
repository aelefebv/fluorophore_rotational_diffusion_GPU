import os.path

from rotational_diffusion.src import components, variables, np
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
               anim_frame_num=None,
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
        anim_subdir = os.path.join(filepath, subdir_name)
        os.makedirs(anim_subdir, exist_ok=True)
        plt.savefig(os.path.join(anim_subdir, f'{anim_frame_num:09}.png'))
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


def time_evolve_and_save_frames(fluorophores, ids, time_step_ns,
                                current_frame_num, current_time_point,
                                remove_ground=True):
    current_frame_num += 1
    if time_step_ns != 0:
        current_time_point = running_time_evolve(fluorophores, time_step_ns, current_time_point)
    if remove_ground:
        fluorophores.delete_fluorophores_in_state('ground')
    save_frame(fluorophores, ids,
               save=True, filepath=output_dir, anim_frame_num=current_frame_num)
    save_frame(fluorophores, ids,
               save=True, filepath=output_dir, anim_frame_num=current_frame_num,
               view_angle=(0, 0), subdir_name='2d_proj_1')
    save_frame(fluorophores, ids,
               save=True, filepath=output_dir, anim_frame_num=current_frame_num,
               view_angle=(0, 90), subdir_name='2d_proj_2')
    return current_frame_num, current_time_point


# setup manually
output_dir_top = r"C:\Users\austin\test_files\julia"
num_molecules = 1E6
start_step_ns = 0.5
start_step_log_ns = numpy.log10(start_step_ns)
end_step_ns = 50
end_step_log_ns = numpy.log10(end_step_ns)

frame_num = 0
time_point = 0
dt = datetime.now().strftime('%Y%m%d%H%M%S')
output_dir = os.path.join(output_dir_top, 'output', dt)
os.makedirs(output_dir)

molecule_properties = components.experiment.MoleculeProperties(
    molecule=variables.molecule_properties.mScarlet,
    num_molecules=num_molecules,
    rotational_diffusion_time=10000 * np.pi,
)
excitation_properties = components.experiment.ExcitationProperties(
    singlet_polarization=(0, 1, 0), singlet_intensity=2,
    crescent_polarization=(1, 0, 0), crescent_intensity=0,
    trigger_polarization=(1, 0, 0), trigger_intensity=3,
    num_triggers=1,
)

# animation frames:
# a. show all (i.e. only ground state) molecules diffusing
ids_to_track = get_ids_to_track(molecule_properties.fluorophores, None, num_molecules)
for _ in range(10):
    frame_num, time_point = time_evolve_and_save_frames(
        molecule_properties.fluorophores, ids_to_track, start_step_ns, frame_num, time_point, remove_ground=False
    )
# generate effective polarization ratios at each time point?
# b. excite to singlet --> ground and singlets
molecule_properties.fluorophores.phototransition(
    'ground', 'singlet',
    intensity=excitation_properties.singlet_intensity,
    polarization_xyz=excitation_properties.singlet_polarization,
)
frame_num, time_point = time_evolve_and_save_frames(
    molecule_properties.fluorophores, ids_to_track, 0, frame_num, time_point, remove_ground=True
)
# c. wait for singlets to decay --> continuous frames (maybe 0.25s resolution?) showing only singlets and triplets
#       up to delay of 25 ns
#       should see singlets disappear and triplets appear
singlet_decay_len_ns = 25
for _ in range(int(singlet_decay_len_ns/start_step_ns)):
    frame_num, time_point = time_evolve_and_save_frames(
        molecule_properties.fluorophores, ids_to_track, start_step_ns, frame_num, time_point, remove_ground=True
    )
# optional crescent selection
if excitation_properties.crescent_intensity > 0:
    molecule_properties.fluorophores.phototransition(
        'triplet', 'singlet',
        intensity=excitation_properties.crescent_intensity,
        polarization_xyz=excitation_properties.crescent_polarization,
    )
# e. Show triplets only, diffusing until some equilibrated timepoint
molecule_properties.fluorophores.delete_fluorophores_in_state('ground')
ids_to_track = get_ids_to_track(molecule_properties.fluorophores, 'triplet', num_molecules)
# - a few frames at the slow timestep
for _ in range(20):
    frame_num, time_point = time_evolve_and_save_frames(
        molecule_properties.fluorophores, ids_to_track, start_step_ns, frame_num, time_point, remove_ground=False
    )
# # - speed up from start step to end step
# triplet_times = numpy.logspace(start_step_log_ns, end_step_log_ns, num=100)
# for _, triplet_time in enumerate(triplet_times):
#     frame_num, time_point = time_evolve_and_save_frames(
#         molecule_properties.fluorophores, ids_to_track, triplet_time, frame_num, time_point, remove_ground=False
#     )
# - let triplets diffuse at fast time steps
for _ in range(500):
    frame_num, time_point = time_evolve_and_save_frames(
        molecule_properties.fluorophores, ids_to_track, end_step_ns, frame_num, time_point, remove_ground=False
    )
subdirs = ['3d', '2d_proj_1', '2d_proj_2']
for subdir in subdirs:
    subdir_path = os.path.join(output_dir, subdir)
    write_gif_from_folder(subdir_path, output_dir, file_name=f'{subdir}.gif')
