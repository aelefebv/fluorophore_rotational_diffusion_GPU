from rotational_diffusion.src import np
from rotational_diffusion.src.components import experiment
from rotational_diffusion.src.variables import molecule_properties, excitation_schemes
from rotational_diffusion.src.utils.base_logger import logger
import time
import os

## User variables
NUM_MOLECULES = 2E07  # Decrease = faster, noisier
# NUM_MOLECULES = 1E06  # Decrease = faster, noisier
EXPERIMENTAL_REPETITIONS = 50  # Decrease = faster, noisier
# EXPERIMENTAL_REPETITIONS = 1  # Decrease = faster, noisier

# Experimental solo variables
fluorophore = molecule_properties.mScarlet
excitation_scheme = excitation_schemes.capture_phosphorescence_decay
singlet_polarization = (0, 1, 0)
crescent_polarization = np.nan
trigger_polarization = np.nan
number_of_triggers = 1
singlet_intensity = 2
crescent_intensity = np.nan
triplet_trigger_intensity = np.nan
collection_time = 5E6  # only collect at 5 ms

## Experimental multi-variables
phosphorescence_lifetimes = [1E4, 2E4, 4E4, 8E4, 16E4, 32E4, 64E4, 128E4]
beads_nm = {'50': 9050, '100': 72400, '200': 579200}  # these get multiplied by pi during the simulation
rotational_diffusion_times = list(beads_nm.values())

# For saving
datetime = time.strftime('%Y%m%d_%H%M%S')
csv_subdir_path = os.path.join('data', f'{datetime}_phospho_lifetime.csv')
csv_path = os.path.join(os.path.dirname(__file__), csv_subdir_path)

# create the path if it doesn't exist
if not os.path.exists(os.path.dirname(csv_path)):
    os.makedirs(os.path.dirname(csv_path))

# Run the multi-variate simulation
for bead_number, rotational_diffusion_time in enumerate(rotational_diffusion_times):
    for phosphorescence_lifetime_num, phosphorescence_lifetime in enumerate(phosphorescence_lifetimes):
        logger.info(f'\nSample \t\t\t\t\t\t{bead_number + 1} of {len(rotational_diffusion_times)}\n'
                    f'Phosphorescence lifetime \t{phosphorescence_lifetime_num + 1} of {len(phosphorescence_lifetimes)}\n')
        molecule_properties = experiment.MoleculeProperties(
            molecule=fluorophore,
            num_molecules=NUM_MOLECULES,
            rotational_diffusion_time=rotational_diffusion_time * np.pi,
            phospho_lifetime=phosphorescence_lifetime,
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
            excitation_scheme=excitation_scheme,
            phosphorescence_collection=True,
        )
        experiment_run[1].add_attributes({'phosphorescence_lifetime': f'{phosphorescence_lifetime}'})
        for experiment_output in experiment_run:
            experiment_run[experiment_output].csv_save(csv_path)
