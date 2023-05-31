import copy
import csv
from dataclasses import dataclass
from datetime import datetime
import os

from rotational_diffusion.src import np, fluorophore  # for GPU-agnosticism
from rotational_diffusion.src.utils.base_logger import logger

## User variables
NUM_MOLECULES = 1E05  # Decrease = faster, noisier
EXPERIMENTAL_REPETITIONS = 4  # Decrease = faster, noisier


## Define our fluorophore
@dataclass(frozen=True)  # for immutability and simplicity
class Fluorescein:
    singlet_lifetime_ns: float = 4
    bleach_rate: float = 0.05


## Create excitation scheme
def run_photoswitch_scheme(fluorophores, laser_properties, collection_time_ns=1E6):
    # Assume continuous bleaching
    # Let's assume a period of 10 ns is roughly continuous
    period_ns = 10
    reps = int(np.floor(collection_time_ns / period_ns))

    # Start collecting photons at t=0
    total_run_time = 0
    collection_start_time = total_run_time

    for rep_num in range(reps):
        logger.info(f'Run time {total_run_time/collection_time_ns*100}%')
        # Delete bleached molecules for computational efficiency, useless on first rep
        fluorophores.delete_fluorophores_in_state('bleached')

        # Excite molecules to singlet state
        fluorophores.phototransition(
            'ground', 'excited',
            intensity=laser_properties.intensity,
            polarization_xyz=laser_properties.polarization,
        )

        # Let singlets decay to ground or bleached state
        fluorophores.time_evolve(period_ns)
        total_run_time += period_ns

    # Stop collecting when experiment is over
    collection_end_time = total_run_time

    # This returned collection value list is for downstream analysis
    collection_time_points = [(1, collection_start_time, collection_end_time)]
    return collection_time_points


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


class Experiment:
    def __init__(self, sample: SampleProperties, excitation_props: LaserProperties, repetitions):
        self.sample = sample
        self.excitation_props = excitation_props
        self.repetitions = repetitions

        self.photons_x_mean = 0
        self.photons_x_std = 0
        self.photons_y_mean = 0
        self.photons_y_std = 0
        self.total_photons_mean = 0
        self.total_photons_std = 0

        self.ratio_xy_mean = None
        self.ratio_xy_std = None

        self.datetime = datetime.now().strftime('%Y%m%d_%H%M%S')

        self.ratio_outputs = []

    def run_experiment(self):
        # Repeat experiments for ratio statistics
        ratio_outputs = []
        photons_x = []
        photons_y = []
        total_photons = []
        for rep_num in range(self.repetitions):
            # Clone original properties since it will change throughout the course of the experiment
            sample_copy = copy.deepcopy(self.sample)

            # Log progress
            rep_percentage = round(rep_num / self.repetitions * 100)
            if rep_percentage % 25 == 0:
                logger.info(f'Experiment repetitions: {int((rep_num / self.repetitions) * 100)}%')

            # Run the pulse scheme
            run_photoswitch_scheme(
                fluorophores=sample_copy.fluorophore_holder,
                laser_properties=self.excitation_props,
            )

            # Get the number of photons emitted in each channel
            counts_tuple = self.get_detector_counts(
                sample_copy.fluorophore_holder,
                'excited', 'ground'
            )
            ratio_outputs.append(counts_tuple[0])
            photons_x.append(counts_tuple[1])
            photons_y.append(counts_tuple[2])
            total_photons.append(counts_tuple[3])

        # Calculate the mean and standard deviation of the arrays
        self.ratio_xy_mean = np.nanmean(np.array(ratio_outputs))
        self.ratio_xy_std = np.nanstd(np.array(ratio_outputs))
        self.photons_x_mean = np.nanmean(np.array(photons_x))
        self.photons_x_std = np.nanstd(np.array(photons_x))
        self.photons_y_mean = np.nanmean(np.array(photons_y))
        self.photons_y_std = np.nanstd(np.array(photons_y))
        self.total_photons_mean = np.nanmean(np.array(total_photons))
        self.total_photons_std = np.nanstd(np.array(total_photons))

    @staticmethod
    def get_detector_counts(fluorophores, from_state, to_state):
        x, y, _, t, = fluorophores.get_xyzt_at_transitions(from_state, to_state)

        p_x, p_y = x ** 2, y ** 2
        r = np.random.uniform(0, 1, size=len(t))
        in_channel_x = (r < p_x)
        in_channel_y = (p_x <= r) & (r < p_x + p_y)
        t_x, t_y = t[in_channel_x], t[in_channel_y]

        photons_x = len(t_x)
        photons_y = len(t_y)
        total_num = photons_x + photons_y

        if len(t_y) <= 0:
            ratio_xy = np.nan
        else:
            ratio_xy = len(t_x) / len(t_y)

        return ratio_xy, photons_x, photons_y, total_num

    def csv_save(self, csv_path):
        # get the list of attributes
        attributes = self.__dict__
        attr_names = list(attributes.keys())
        attr_values = []
        for value in list(attributes.values()):
            attr_values.append(value)

        laser_attributes = self.excitation_props.__dict__
        for name, value in laser_attributes.items():
            attr_names.append(name)
            attr_values.append(value)

        for name, value in self.sample.__dict__.items():
            attr_names.append(f'sample_{name}')
            attr_values.append(value)

        # Check if the directory exists
        if not os.path.exists(os.path.dirname(csv_path)):
            os.makedirs(os.path.dirname(csv_path))

        # Check if the file exists
        if not os.path.exists(csv_path):
            with open(csv_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile, delimiter=',')
                # Write the header row
                writer.writerow(attr_names)

        # Open the file for appending
        with open(csv_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(attr_values)


# Setup and run the experiment
def run():
    ## Experimental solo variables
    fluorophore_molecule = Fluorescein()

    ## Experimental multi-variables
    # ref: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3823292/
    ab_binding = {'ab': 250, 'ab-m': 260, 'ab-agg1': 417, 'ab-sol': 666,
                  'ab-proto': 2000}  # these get multiplied by pi during the simulation
    rotational_diffusion_times = list(ab_binding.values())

    ## For saving
    date = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_subdir_path = os.path.join('data', f'{date}_photobleach.csv')
    csv_path = os.path.join(os.path.dirname(__file__), csv_subdir_path)

    ## Create state info
    # On state is essentially forever in this time regime
    ground_state = fluorophore.ElectronicState('ground')
    bleached_state = fluorophore.ElectronicState('bleached')
    # Excited state emits a photon then turns goes to ground or bleaches
    excited_transitions = ['ground', 'bleached']
    excited_transition_probabilities = [1 - fluorophore_molecule.bleach_rate, fluorophore_molecule.bleach_rate]
    excited_state = fluorophore.ElectronicState(
        'excited',
        lifetime=fluorophore_molecule.singlet_lifetime_ns,
        transition_states=excited_transitions,
        probabilities=excited_transition_probabilities,
    )
    # Add states to the fluorophore
    state_info = fluorophore.PossibleStates(ground_state)
    state_info.add_state(bleached_state)
    state_info.add_state(excited_state)

    ## Define some excitation properties:
    excitation_intensities = [0.002, 0.010, 0.05, 0.25, 1.25, 6.25]
    excitation_polarization = (1, 0, 0)

    # Run the multi-variate simulation
    for excitation_num, excitation_intensity in enumerate(excitation_intensities):
        for sample_num, rotational_diffusion_time in enumerate(rotational_diffusion_times):
            logger.info(f'\nSample \t\t\t\t\t\t{sample_num + 1} of {len(rotational_diffusion_times)}\n'
                        f'Excitation intensity \t\t{excitation_num + 1} of {len(excitation_intensities)}')
            sample = SampleProperties(
                fluorescent_molecule=fluorophore_molecule,
                num_molecules=NUM_MOLECULES,
                rdt=rotational_diffusion_time,
                fluorophore_state_info=state_info,
            )

            excitation_laser = LaserProperties(
                intensity=excitation_intensity,
                polarization=excitation_polarization,
            )

            experiment = Experiment(
                sample=sample,
                excitation_props=excitation_laser,
                repetitions=EXPERIMENTAL_REPETITIONS
            )
            experiment.run_experiment()
            experiment.csv_save(csv_path)


if __name__ == '__main__':
    run()
