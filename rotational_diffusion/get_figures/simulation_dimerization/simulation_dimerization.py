import copy
import csv
from dataclasses import dataclass
from datetime import datetime
import os

from rotational_diffusion.src import np, fluorophore  # for GPU-agnosticism
from rotational_diffusion.src.utils.base_logger import logger       # for logging progress

## User variables
NUM_MOLECULES = 2E07              # default 2E07,     Decrease = faster, noisier
EXPERIMENTAL_REPETITIONS = 50     # default 50,        Decrease = faster, noisier


## Define our fluorophore's properties
@dataclass(frozen=True)  # for immutability and simplicity
class mScarlet:
    singlet_lifetime_ns: float = 3
    triplet_lifetime_ns: float = 5E05
    triplet_quantum_yield: float = 0.01


## Create excitation scheme
def run_dimerization_scheme(fluorophores, fluorophore_properties, laser_properties, collection_time_point_ns):
    total_run_time = 0

    # excite molecules to singlet state
    fluorophores.phototransition(
        'ground', 'singlet',
        intensity=laser_properties.excitation_laser.intensity,
        polarization_xyz=laser_properties.excitation_laser.polarization,
    )

    # let excited molecules go to ground or triplet (10x singlet lifetime)
    decay_time = fluorophore_properties.singlet_lifetime_ns * 10
    fluorophores.time_evolve(decay_time)
    total_run_time += decay_time
    # delete anything in the ground state, which is now useless
    fluorophores.delete_fluorophores_in_state('ground')

    # let molecules diffuse until the trigger
    fluorophores.time_evolve(collection_time_point_ns)
    total_run_time += collection_time_point_ns
    fluorophores.delete_fluorophores_in_state('ground')

    # collect photons only after the trigger
    collection_start_time = total_run_time

    # trigger triplets back to singlets
    fluorophores.phototransition(
        'triplet', 'singlet',
        intensity=laser_properties.trigger_laser.intensity,
        polarization_xyz=laser_properties.trigger_laser.polarization
    )

    # let singlets decay to ground
    fluorophores.time_evolve(decay_time)
    total_run_time += decay_time
    fluorophores.delete_fluorophores_in_state('ground')

    # collect photons until the end of the experiment
    collection_end_time = total_run_time

    # This returned collection value list is for downstream time-gated analysis
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


## Store the excitation properties of both lasers
class ExcitationProperties:
    def __init__(self, excitation_laser: LaserProperties, trigger_laser: LaserProperties):
        self.excitation_laser = excitation_laser
        self.trigger_laser = trigger_laser


class Experiment:
    def __init__(self,
                 sample: SampleProperties,
                 excitation_props: ExcitationProperties,
                 collection_time_point_ns: float,
                 repetitions):
        self.sample = sample
        self.excitation_props = excitation_props
        self.repetitions = repetitions

        self.collection_time_point = collection_time_point_ns

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
            collection_times = run_dimerization_scheme(
                fluorophores=sample_copy.fluorophore_holder,
                fluorophore_properties=self.sample.fluorescent_molecule,
                laser_properties=self.excitation_props,
                collection_time_point_ns=self.collection_time_point,
            )

            # Get the number of photons emitted in each channel
            counts_tuple = self.get_detector_counts(
                sample_copy.fluorophore_holder,
                'singlet', 'ground',
                collection_times[0][1:]
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
    def get_detector_counts(fluorophores, from_state, to_state, collection_times):
        x, y, _, t, = fluorophores.get_xyzt_at_transitions(from_state, to_state)

        t_gated = t[(t >= collection_times[0]) & (t <= collection_times[1])]
        x_gated = x[(t >= collection_times[0]) & (t <= collection_times[1])]
        y_gated = y[(t >= collection_times[0]) & (t <= collection_times[1])]

        p_x, p_y = x_gated ** 2, y_gated ** 2
        r = np.random.uniform(0, 1, size=len(t_gated))
        in_channel_x = (r < p_x)
        in_channel_y = (p_x <= r) & (r < p_x + p_y)
        t_x, t_y = t_gated[in_channel_x], t_gated[in_channel_y]

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

        excitation_attr = self.excitation_props.excitation_laser.__dict__
        for name, value in excitation_attr.items():
            attr_names.append(f'excitation_{name}')
            attr_values.append(value)

        trigger_attr = self.excitation_props.trigger_laser.__dict__
        for name, value in trigger_attr.items():
            attr_names.append(f'trigger_{name}')
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
    ## Fluorophore properties
    fluorophore_molecule = mScarlet()

    ## Molecule of interest properties
    # ref: https://febs.onlinelibrary.wiley.com/doi/full/10.1016/S0014-5793%2898%2901425-2
    mers = {'monomer': 250, 'dimer': 500}  # these get multiplied by pi during the simulation
    # these get multiplied by pi during the simulation
    rotational_diffusion_times = list(mers.values())

    ## For saving
    date = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_subdir_path = os.path.join('data', f'{date}_dimerization.csv')
    csv_path = os.path.join(os.path.dirname(__file__), csv_subdir_path)

    ## Create state info
    ground_state = fluorophore.ElectronicState('ground')
    # Excited state emits a photon then turns off, we assume it doesn't die
    singlet_transitions = ['ground', 'triplet']
    singlet_transition_probabilities = [1 - fluorophore_molecule.triplet_quantum_yield,
                                        fluorophore_molecule.triplet_quantum_yield]
    singlet_state = fluorophore.ElectronicState(
        'singlet',
        lifetime=fluorophore_molecule.singlet_lifetime_ns,
        transition_states=singlet_transitions,
        probabilities=singlet_transition_probabilities,
    )
    triplet_state = fluorophore.ElectronicState(
        'triplet', lifetime=fluorophore_molecule.triplet_lifetime_ns,
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

    ## Define some collection time points
    collection_times_ns = np.linspace(50, 5000, num=100).tolist()

    # Run the multi-variate simulation
    for sample_num, rotational_diffusion_time in enumerate(rotational_diffusion_times):
        for collection_num, collection_time_point_ns in enumerate(collection_times_ns):
            logger.info(f'\nSample \t\t\t\t\t\t{sample_num + 1} of {len(rotational_diffusion_times)}\n'
                        f'Collection \t\t\t\t{collection_num + 1} of {len(collection_times_ns)}\n')
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
            trigger_laser = LaserProperties(
                intensity=trigger_intensity,
                polarization=trigger_polarization,
            )
            excitation_properties = ExcitationProperties(
                excitation_laser=excitation_laser,
                trigger_laser=trigger_laser,
            )

            experiment = Experiment(
                sample=sample,
                excitation_props=excitation_properties,
                collection_time_point_ns=collection_time_point_ns,
                repetitions=EXPERIMENTAL_REPETITIONS
            )
            experiment.run_experiment()
            experiment.csv_save(csv_path)


if __name__ == '__main__':
    run()
