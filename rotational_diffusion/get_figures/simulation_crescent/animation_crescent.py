import numpy
import os
from dataclasses import dataclass
from datetime import datetime

from rotational_diffusion.src import np

from rotational_diffusion.src.components import fluorophore
from rotational_diffusion.src.utils import animating

## User variables
NUM_MOLECULES = 1E06  # Decrease = faster, noisier
rotational_diffusion_times = [7249, 24465, 113263, 906106]

## Define our fluorophore's properties
@dataclass(frozen=True)  # for immutability and simplicity
class mScarlet:
    singlet_lifetime_ns: float = 3
    triplet_lifetime_ns: float = 5E05
    triplet_quantum_yield: float = 0.01


## Store the sample properties
class SampleProperties:
    def __init__(self, fluorescent_molecule, num_molecules, rdt, fluorophore_state_info):
        self.fluorescent_molecule = fluorescent_molecule
        self.num_molecules = num_molecules
        self.rdt = rdt
        self.rdt_unpied = self.rdt / np.pi
        self.state_info = fluorophore_state_info
        self.fluorophore_holder = fluorophore.FluorophoreCollection(
            num_molecules=self.num_molecules,
            state_info=self.state_info,
            rot_diffusion_time=self.rdt
        )


## Store the laser properties
class LaserProperties:
    def __init__(self, intensity, polarization):
        self.intensity = intensity
        self.polarization = polarization


## Store the excitation properties of both lasers
class ExcitationProperties:
    def __init__(self,
                 excitation_laser: LaserProperties,
                 trigger_laser: LaserProperties,
                 crescent_laser: LaserProperties):
        self.excitation_laser = excitation_laser
        self.trigger_laser = trigger_laser
        self.crescent_laser = crescent_laser


## Define our fluorophore
fluorescent_molecule = mScarlet

## Create state info
ground_state = fluorophore.ElectronicState('ground')
# Excited state emits a photon then turns off, we assume it doesn't die
singlet_transitions = ['ground', 'triplet']
singlet_transition_probabilities = [1 - fluorescent_molecule.triplet_quantum_yield,
                                    fluorescent_molecule.triplet_quantum_yield]
singlet_state = fluorophore.ElectronicState(
    'singlet',
    lifetime=fluorescent_molecule.singlet_lifetime_ns,
    transition_states=singlet_transitions,
    probabilities=singlet_transition_probabilities,
)
triplet_state = fluorophore.ElectronicState(
    'triplet', lifetime=fluorescent_molecule.triplet_lifetime_ns,
    transition_states='ground',
    probabilities=1,
)
# Add states to the fluorophore
state_info = fluorophore.PossibleStates(ground_state)
state_info.add_state(singlet_state)
state_info.add_state(triplet_state)

## Define some excitation properties:
excitation_intensity = 2
excitation_polarization = (0, 1, 0)
trigger_intensity = 0.25
trigger_polarization = (1, 0, 0)
crescent_intensity = 4
crescent_polarization = (1, 0, 0)

# For saving
date = datetime.now().strftime('%Y%m%d_%H%M%S')
output_dir_top = os.path.dirname(__file__)

for rotational_diffusion_time in rotational_diffusion_times:
    output_dir = os.path.join(output_dir_top, 'images', f'{date}-{rotational_diffusion_time}rdt')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Animation variables
    start_step_ns = 0.5
    start_step_log_ns = numpy.log10(start_step_ns)
    end_step_ns = 50
    end_step_log_ns = numpy.log10(end_step_ns)
    slow_triplet_frames = 20
    fast_triplet_frames = 500

    sample = SampleProperties(
        fluorescent_molecule=fluorescent_molecule,
        num_molecules=NUM_MOLECULES,
        rdt=rotational_diffusion_time,
        fluorophore_state_info=state_info,
    )

    excitation_laser = LaserProperties(
        intensity=excitation_intensity,
        polarization=excitation_polarization,
    )
    trigger_laser = LaserProperties(
        intensity=trigger_intensity,
        polarization=trigger_polarization,
    )
    crescent_laser = LaserProperties(
        intensity=crescent_intensity,
        polarization=crescent_polarization,
    )
    laser_properties = ExcitationProperties(
        excitation_laser=excitation_laser,
        trigger_laser=trigger_laser,
        crescent_laser=crescent_laser,
    )

    # Initialize
    frame_num = 0
    time_point = 0

    ## Run animation
    # Series A: Show all (i.e. only ground state) molecules diffusing
    ids_to_track = animating.get_ids_to_track(sample.fluorophore_holder, None, NUM_MOLECULES)
    for _ in range(10):
        frame_num, time_point = animating.time_evolve_and_save_frames(
            sample, ids_to_track, start_step_ns, frame_num, time_point, output_dir,
            remove_ground=False
        )

    # Series B: Excite to singlet
    sample.fluorophore_holder.phototransition(
        'ground', 'singlet',
        intensity=laser_properties.excitation_laser.intensity,
        polarization_xyz=laser_properties.excitation_laser.polarization,
    )
    frame_num, time_point = animating.time_evolve_and_save_frames(
        sample, ids_to_track, 0, frame_num, time_point, output_dir, remove_ground=True
    )

    # Series C: Show singlet decay up to delay of 25 ns. Should see singlets disappear and triplets appear
    singlet_decay_len_ns = 25
    for _ in range(int(singlet_decay_len_ns/start_step_ns)):
        frame_num, time_point = animating.time_evolve_and_save_frames(
            sample, ids_to_track, start_step_ns, frame_num, time_point, output_dir,
            remove_ground=True
        )

    # Crescent selection.
    if laser_properties.crescent_laser.intensity > 0:
        sample.fluorophore_holder.phototransition(
            'triplet', 'singlet',
            intensity=laser_properties.crescent_laser.intensity,
            polarization_xyz=laser_properties.crescent_laser.polarization,
        )

    # Series D: Show triplets only, diffusing until some equilibrated timepoint
    sample.fluorophore_holder.delete_fluorophores_in_state('ground')
    ids_to_track = animating.get_ids_to_track(sample.fluorophore_holder, 'triplet', NUM_MOLECULES)
    # First, show a few frames at the slow timestep
    for _ in range(slow_triplet_frames):
        frame_num, time_point = animating.time_evolve_and_save_frames(
            sample, ids_to_track, start_step_ns, frame_num, time_point, output_dir,
            remove_ground=False
        )

    # Finally, let triplets diffuse at fast timesteps
    for _ in range(fast_triplet_frames):
        frame_num, time_point = animating.time_evolve_and_save_frames(
            sample, ids_to_track, end_step_ns, frame_num, time_point, output_dir,
            remove_ground=False
        )

    # Save the animation for each projection angle
    subdirs = ['3d', '2d_proj_1', '2d_proj_2', '2d_proj_3']
    for subdir in subdirs:
        subdir_path = os.path.join(output_dir, subdir)
        animating.write_gif_from_folder(subdir_path, output_dir, file_name=f'{subdir}.gif')

