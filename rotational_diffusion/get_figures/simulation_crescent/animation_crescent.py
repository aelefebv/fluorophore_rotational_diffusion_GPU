import numpy
import os
from rotational_diffusion.src.visualization import animating
from rotational_diffusion.src import components, variables, np
import time

## User variables
NUM_MOLECULES = 1E06  # Decrease = faster, noisier
beads_nm = {'40': 4630, '60': 15640, '100': 72400, '200': 579200}  # these get multiplied by pi during the simulation
rotational_diffusion_times = beads_nm['100']  # choose from beads_nm, which one to animate


# Experimental solo variables
fluorophore = variables.molecule_properties.mScarlet
triplet_trigger_intensity = 0.25
singlet_polarization = (0, 1, 0)
crescent_polarization = (1, 0, 0)
trigger_polarization = (1, 0, 0)
number_of_triggers = 1

## Experimental multi-variables
# 100 nm bead rotational diffusion time is from STARSS paper. The rest are relative to that.
collection_times_ns = np.linspace(10000, 1E6, num=100).tolist()
crescent_intensities = [0]
singlet_intensities = [2]

# For saving
datetime = time.strftime('%Y%m%d_%H%M%S')
output_dir_top = os.path.dirname(__file__)
output_dir = os.path.join(output_dir_top, 'images', datetime)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

start_step_ns = 0.5
start_step_log_ns = numpy.log10(start_step_ns)
end_step_ns = 50
end_step_log_ns = numpy.log10(end_step_ns)

frame_num = 0
time_point = 0

molecule_properties = components.experiment.MoleculeProperties(
    molecule=fluorophore,
    num_molecules=NUM_MOLECULES,
    rotational_diffusion_time=2500 * np.pi,
)
excitation_properties = components.experiment.ExcitationProperties(
    singlet_polarization=(0, 1, 0), singlet_intensity=2,
    crescent_polarization=(1, 0, 0), crescent_intensity=0,
    trigger_polarization=(1, 0, 0), trigger_intensity=3,
    num_triggers=1,
)

# animation frames:
# a. show all (i.e. only ground state) molecules diffusing
ids_to_track = animating.get_ids_to_track(molecule_properties.fluorophores, None, NUM_MOLECULES)
for _ in range(10):
    frame_num, time_point = animating.time_evolve_and_save_frames(
        molecule_properties.fluorophores, ids_to_track, start_step_ns, frame_num, time_point, remove_ground=False
    )
# generate effective polarization ratios at each time point?
# b. excite to singlet --> ground and singlets
molecule_properties.fluorophores.phototransition(
    'ground', 'singlet',
    intensity=excitation_properties.singlet_intensity,
    polarization_xyz=excitation_properties.singlet_polarization,
)
frame_num, time_point = animating.time_evolve_and_save_frames(
    molecule_properties.fluorophores, ids_to_track, 0, frame_num, time_point, remove_ground=True
)
# c. wait for singlets to decay --> continuous frames (maybe 0.25s resolution?) showing only singlets and triplets
#       up to delay of 25 ns
#       should see singlets disappear and triplets appear
singlet_decay_len_ns = 25
for _ in range(int(singlet_decay_len_ns/start_step_ns)):
    frame_num, time_point = animating.time_evolve_and_save_frames(
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
ids_to_track = animating.get_ids_to_track(molecule_properties.fluorophores, 'triplet', NUM_MOLECULES)
# - a few frames at the slow timestep
for _ in range(20):
    frame_num, time_point = animating.time_evolve_and_save_frames(
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
    frame_num, time_point = animating.time_evolve_and_save_frames(
        molecule_properties.fluorophores, ids_to_track, end_step_ns, frame_num, time_point, remove_ground=False
    )
subdirs = ['3d', '2d_proj_1', '2d_proj_2', '2d_proj_3']
for subdir in subdirs:
    subdir_path = os.path.join(output_dir, subdir)
    animating.write_gif_from_folder(subdir_path, output_dir, file_name=f'{subdir}.gif')


# todo save parameters (or entire code) used for each run in a text file