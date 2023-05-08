from rotational_diffusion.src import np
from rotational_diffusion.src.components import experiment
from rotational_diffusion.src.variables import molecule_properties, excitation_schemes
from rotational_diffusion.src.utils.base_logger import logger
import time
import os

## User variables
# NUM_MOLECULES = 2E07  # Decrease = faster, noisier
NUM_MOLECULES = 1E06  # Decrease = faster, noisier
EXPERIMENTAL_REPETITIONS = 3  # Decrease = faster, noisier
# EXPERIMENTAL_REPETITIONS = 1  # Decrease = faster, noisier

# Experimental solo variables
fluorophore = molecule_properties.mScarlet
excitation_scheme = excitation_schemes.cw
singlet_polarization = (0, 1, 0)
crescent_polarization = np.nan
trigger_polarization = 'circular'
number_of_triggers = 1
crescent_intensity = np.nan
collection_time = 5E6  # only collect at 5 ms
singlet_intensity = 2
rotational_diffusion_time = 113263  # 100 nm bead
cw_delay_ns = 100 # ns

## Experimental multi-variables
triplet_trigger_intensities = [0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56, 5.12, 10.24]

# For saving
datetime = time.strftime('%Y%m%d_%H%M%S')
csv_subdir_path = os.path.join('data', f'{datetime}_cw_lifetime.csv')
csv_path = os.path.join(os.path.dirname(__file__), csv_subdir_path)

# create the path if it doesn't exist
if not os.path.exists(os.path.dirname(csv_path)):
    os.makedirs(os.path.dirname(csv_path))

# Run the multi-variate simulation
for triplet_trigger_intensity_num, triplet_trigger_intensity in enumerate(triplet_trigger_intensities):
    logger.info(f'\nTriplet trigger intensity \t{triplet_trigger_intensity_num + 1} of {len(triplet_trigger_intensities)}\n')
    molecule_properties = experiment.MoleculeProperties(
        molecule=fluorophore,
        num_molecules=NUM_MOLECULES,
        rotational_diffusion_time=rotational_diffusion_time * np.pi,
    )
    excitation_properties = experiment.ExcitationProperties(
        singlet_polarization=singlet_polarization, singlet_intensity=singlet_intensity,
        crescent_polarization=crescent_polarization, crescent_intensity=crescent_intensity,
        trigger_polarization=trigger_polarization, trigger_intensity=triplet_trigger_intensity,
        num_triggers=number_of_triggers, cw_delay=cw_delay_ns
    )
    experiment_run = experiment.run_experiment(
        molecule_properties, excitation_properties,
        collection_time_point_ns=collection_time,
        repetitions=EXPERIMENTAL_REPETITIONS,
        excitation_scheme=excitation_scheme,
        # trigger_collection=True,
    )
    for experiment_num, experiment_output in enumerate(experiment_run):
        experiment_run[experiment_output].csv_save(csv_path)
