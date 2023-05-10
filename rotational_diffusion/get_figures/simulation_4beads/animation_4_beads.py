import numpy
import os
from rotational_diffusion.src.visualization import animating
from rotational_diffusion.src import components, variables, np
import time

## User variables
NUM_MOLECULES = 1E06  # Decrease = faster, noisier
rotational_diffusion_times = [7249, 24465, 113263, 906106]  # choose from beads_nm, which one to animate

# Experimental solo variables
fluorophore = variables.molecule_properties.mScarlet
triplet_trigger_intensity = 0.25
singlet_polarization = (0, 1, 0)
crescent_polarization = (1, 0, 0)
trigger_polarization = (1, 0, 0)
number_of_triggers = 1
crescent_intensities = 0
singlet_intensity = 2
trigger_intensity = 0.25  # shouldn't actually affect anything.

# For saving
datetime = time.strftime('%Y%m%d_%H%M%S')
output_dir_top = os.path.dirname(__file__)

for rotational_diffusion_time in rotational_diffusion_times:
    output_dir = os.path.join(output_dir_top, 'images', f'{datetime}-{rotational_diffusion_time}rdt')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Animation variables
    start_step_ns = 0.5
    start_step_log_ns = numpy.log10(start_step_ns)
    end_step_ns = 50
    end_step_log_ns = numpy.log10(end_step_ns)
    slow_triplet_frames = 20
    fast_triplet_frames = 500


    # Set up the experiment
    molecule_properties = components.experiment.MoleculeProperties(
        molecule=fluorophore,
        num_molecules=NUM_MOLECULES,
        rotational_diffusion_time=rotational_diffusion_time * np.pi,
    )
    excitation_properties = components.experiment.ExcitationProperties(
        singlet_polarization=singlet_polarization, singlet_intensity=singlet_intensity,
        crescent_polarization=crescent_polarization, crescent_intensity=crescent_intensities,
        trigger_polarization=trigger_polarization, trigger_intensity=trigger_intensity,
        num_triggers=number_of_triggers,
    )
    # Initialize
    frame_num = 0
    time_point = 0

    ## Run animation
    # Series A: Show all (i.e. only ground state) molecules diffusing
    ids_to_track = animating.get_ids_to_track(molecule_properties.fluorophores, None, NUM_MOLECULES)
    for _ in range(10):
        frame_num, time_point = animating.time_evolve_and_save_frames(
            molecule_properties, ids_to_track, start_step_ns, frame_num, time_point, output_dir,
            remove_ground=False
        )

    # Series B: Excite to singlet
    molecule_properties.fluorophores.phototransition(
        'ground', 'singlet',
        intensity=excitation_properties.singlet_intensity,
        polarization_xyz=excitation_properties.singlet_polarization,
    )
    frame_num, time_point = animating.time_evolve_and_save_frames(
        molecule_properties, ids_to_track, 0, frame_num, time_point, output_dir, remove_ground=True
    )

    # Series C: Show singlet decay up to delay of 25 ns. Should see singlets disappear and triplets appear
    singlet_decay_len_ns = 25
    for _ in range(int(singlet_decay_len_ns/start_step_ns)):
        frame_num, time_point = animating.time_evolve_and_save_frames(
            molecule_properties, ids_to_track, start_step_ns, frame_num, time_point, output_dir,
            remove_ground=True
        )

    # Optional: Crescent selection.
    if excitation_properties.crescent_intensity > 0:
        molecule_properties.fluorophores.phototransition(
            'triplet', 'singlet',
            intensity=excitation_properties.crescent_intensity,
            polarization_xyz=excitation_properties.crescent_polarization,
        )

    # Series D: Show triplets only, diffusing until some equilibrated timepoint
    molecule_properties.fluorophores.delete_fluorophores_in_state('ground')
    ids_to_track = animating.get_ids_to_track(molecule_properties.fluorophores, 'triplet', NUM_MOLECULES)
    # First, show a few frames at the slow timestep
    for _ in range(slow_triplet_frames):
        frame_num, time_point = animating.time_evolve_and_save_frames(
            molecule_properties, ids_to_track, start_step_ns, frame_num, time_point, output_dir,
            remove_ground=False
        )

    # # Optionally, slowly speed up from start step to end step
    # triplet_times = numpy.logspace(start_step_log_ns, end_step_log_ns, num=100)
    # for _, triplet_time in enumerate(triplet_times):
    #     frame_num, time_point = time_evolve_and_save_frames(
    #         molecule_properties.fluorophores, ids_to_track, triplet_time, frame_num, time_point, remove_ground=False
    #     )

    # Finally, let triplets diffuse at fast timesteps
    for _ in range(fast_triplet_frames):
        frame_num, time_point = animating.time_evolve_and_save_frames(
            molecule_properties, ids_to_track, end_step_ns, frame_num, time_point, output_dir,
            remove_ground=False
        )

    # Save the animation for each projection angle
    subdirs = ['3d', '2d_proj_1', '2d_proj_2', '2d_proj_3']
    for subdir in subdirs:
        subdir_path = os.path.join(output_dir, subdir)
        animating.write_gif_from_folder(subdir_path, output_dir, file_name=f'{subdir}.gif')
