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


def save_frame(molecule_properties, ids,
               save=True, filepath=None,
               anim_frame_num=None,
               projection_type='3d',
               view_angle=(25, -8),
               subdir_name='3d',
               time=None):

    alpha_0 = 0.01  # ground state opacity
    alpha_1 = 0.5  # triplet state opacity
    alpha_2 = 0.25  # singlet state opacity

    idxs = np.where(np.isin(molecule_properties.fluorophores.id, ids))
    o = molecule_properties.fluorophores.orientations
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
        # Hide X and Y axes label marks
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

        # Hide X and Y axes tick marks
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        if time is not None:
            formatted_time = f"{(time/1000):0>3.3f} us"
            ax.set_title(formatted_time)

    # haven't actually used this for anything yet. Tried it out and didn't like the look...
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
    # need to keep track of time for the fluorophores
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
    # Call ffmpeg to generate gif from pngs. Make the color palette to create nice figures.
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


def time_evolve_and_save_frames(molecule_properties, ids, time_step_ns,
                                current_frame_num, current_time_point, output_dir,
                                remove_ground=True):
    current_frame_num += 1
    if time_step_ns != 0:
        current_time_point = running_time_evolve(molecule_properties.fluorophores, time_step_ns, current_time_point)
    if remove_ground:
        molecule_properties.fluorophores.delete_fluorophores_in_state('ground')
    save_frame(molecule_properties, ids,
               save=True, filepath=output_dir, anim_frame_num=current_frame_num, time=current_time_point)
    save_frame(molecule_properties, ids,
               save=True, filepath=output_dir, anim_frame_num=current_frame_num,
               view_angle=(0, 0), subdir_name='2d_proj_1', time=current_time_point)
    save_frame(molecule_properties, ids,
               save=True, filepath=output_dir, anim_frame_num=current_frame_num,
               view_angle=(0, 90), subdir_name='2d_proj_2', time=current_time_point)
    save_frame(molecule_properties, ids,
               save=True, filepath=output_dir, anim_frame_num=current_frame_num,
               view_angle=(90, 0), subdir_name='2d_proj_3', time=current_time_point)
    return current_frame_num, current_time_point
