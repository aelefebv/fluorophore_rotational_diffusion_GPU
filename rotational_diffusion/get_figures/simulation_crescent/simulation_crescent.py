from rotational_diffusion.src import np
from rotational_diffusion.src.components import experiment
from rotational_diffusion.src.variables import molecule_properties, excitation_schemes
from rotational_diffusion.src.utils.base_logger import logger
import time
import os

## User variables
NUM_MOLECULES = 2E07  # Decrease = faster, noisier
EXPERIMENTAL_REPETITIONS = 5  # Decrease = faster, noisier

# Experimental solo variables
fluorophore = molecule_properties.mScarlet
triplet_trigger_intensity = 0.25
singlet_polarization = (0, 1, 0)
crescent_polarization = (1, 0, 0)
trigger_polarization = (1, 0, 0)
number_of_triggers = 1

## Experimental multi-variables
beads_nm = {'40': 7249, '60': 24465, '100': 113263, '200': 906106}  # these get multiplied by pi during the simulation
rotational_diffusion_times = list(beads_nm.values())
collection_times_ns = np.linspace(10000, 1E6, num=100).tolist()
crescent_intensities = [0, 1, 2, 4, 8]
singlet_intensities = [2]

# For saving
datetime = time.strftime('%Y%m%d_%H%M%S')
csv_subdir_path = os.path.join('data', f'{datetime}_crescent_beads.csv')
csv_path = os.path.join(os.path.dirname(__file__), csv_subdir_path)

# Run the multi-variate simulation
for singlet_number, singlet_intensity in enumerate(singlet_intensities):
    for crescent_number, crescent_intensity in enumerate(crescent_intensities):
        for bead_number, rotational_diffusion_time in enumerate(rotational_diffusion_times):
            for collection_number, collection_time in enumerate(collection_times_ns):
                collection_time = int(collection_time)
                logger.info(f'\nCollection \t\t\t\t\t{collection_number + 1} of {len(collection_times_ns)}\n'
                            f'Sample \t\t\t\t\t\t{bead_number + 1} of {len(rotational_diffusion_times)}\n'
                            f'Crescent intensity \t\t\t{crescent_number + 1} of {len(crescent_intensities)}\n'
                            f'Singlet intensity \t\t\t{singlet_number + 1} of {len(singlet_intensities)}')
                molecule_properties = experiment.MoleculeProperties(
                    molecule=fluorophore,
                    num_molecules=NUM_MOLECULES,
                    rotational_diffusion_time=rotational_diffusion_time * np.pi,
                )
                excitation_properties = experiment.ExcitationProperties(
                    singlet_polarization=singlet_polarization, singlet_intensity=singlet_intensity,
                    crescent_polarization=crescent_polarization, crescent_intensity=crescent_intensity,
                    trigger_polarization=trigger_polarization, trigger_intensity=triplet_trigger_intensity,
                    num_triggers=number_of_triggers,
                )
                experiment_run = experiment.run_experiment(
                    molecule_properties, excitation_properties,
                    collection_time_point_ns=collection_time,
                    repetitions=EXPERIMENTAL_REPETITIONS,
                    excitation_scheme=excitation_schemes.gentle_check,
                )
                for experiment_output in experiment_run:
                    experiment_run[experiment_output].csv_save(csv_path)
