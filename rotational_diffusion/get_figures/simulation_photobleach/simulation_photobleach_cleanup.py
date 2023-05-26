from rotational_diffusion.src import np
from rotational_diffusion.src.components import experiment
from rotational_diffusion.src.variables import molecule_properties, excitation_schemes
from rotational_diffusion.src.utils.base_logger import logger
import time
import os

## User variables
NUM_MOLECULES = 2E07  # Decrease = faster, noisier
EXPERIMENTAL_REPETITIONS = 4  # Decrease = faster, noisier

# Experimental solo variables
fluorophore = molecule_properties.Fluorescein
singlet_polarization = (0, 1, 0)

## Experimental multi-variables
# ref: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3823292/
ab_binding = {'ab': 250, 'ab-m': 260, 'ab-agg1': 417, 'ab-sol': 666, 'ab-proto': 2000}  # these get multiplied by pi during the simulation
rotational_diffusion_times = list(ab_binding.values())
singlet_intensities = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 2]
# bleach_rates = [0.01, 0.025, 0.05, 0.075, 0.1]
bleach_rates = [0.05]

# For saving
datetime = time.strftime('%Y%m%d_%H%M%S')
csv_subdir_path = os.path.join('data', f'{datetime}_photobleach.csv')
csv_path = os.path.join(os.path.dirname(__file__), csv_subdir_path)

# Run the multi-variate simulation
for singlet_number, singlet_intensity in enumerate(singlet_intensities):
    for bead_number, rotational_diffusion_time in enumerate(rotational_diffusion_times):
        for bleach_num, bleach_rate in enumerate(bleach_rates):
            logger.info(f'\nSample \t\t\t\t\t\t{bead_number + 1} of {len(rotational_diffusion_times)}\n'
                        f'Bleach rate \t\t\t\t{bleach_num + 1} of {len(bleach_rates)}\n'
                        f'Singlet intensity \t\t\t{singlet_number + 1} of {len(singlet_intensities)}')
            molecule_properties = experiment.MoleculeProperties(
                molecule=fluorophore,
                num_molecules=NUM_MOLECULES,
                rotational_diffusion_time=rotational_diffusion_time * np.pi,
                triplet=False,
                photobleach=True,
                photobleach_rate=bleach_rate,
            )
            excitation_properties = experiment.ExcitationProperties(
                singlet_polarization=singlet_polarization, singlet_intensity=singlet_intensity,
            )
            experiment_run = experiment.run_experiment(
                molecule_properties, excitation_properties,
                repetitions=EXPERIMENTAL_REPETITIONS,
                excitation_scheme=excitation_schemes.photobleach,
            )
            for experiment_output in experiment_run:
                experiment_run[experiment_output].csv_save(csv_path)
