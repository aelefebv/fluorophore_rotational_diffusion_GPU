import csv
import os

import matplotlib.pyplot as plt

from rotational_diffusion.src.utils.base_logger import logger
from rotational_diffusion.src import utils, components, variables, np

from datetime import datetime


class MoleculeProperties:
    def __init__(self, molecule, num_molecules, rotational_diffusion_time, triplet=True):
        self.molecule = molecule
        self.num_molecules = num_molecules
        self.rotational_diffusion_time = rotational_diffusion_time
        self.fluorophore = self.molecule(self.rotational_diffusion_time)

        if triplet:
            state_info_function = utils.state_info_creator.create_triplet_state_info
        else:
            state_info_function = utils.state_info_creator.create_singlet_state_info
        self.state_info = state_info_function(self.fluorophore)
        self.fluorophores = components.fluorophore.FluorophoreCollection(
            num_molecules=self.num_molecules,
            state_info=self.state_info,
            rot_diffusion_time=self.rotational_diffusion_time
        )


class ExcitationProperties:
    def __init__(self, singlet_polarization, singlet_intensity,
                 crescent_polarization, crescent_intensity,
                 trigger_polarization, trigger_intensity):

        self.singlet_polarization = singlet_polarization
        self.singlet_intensity = singlet_intensity

        self.crescent_polarization = crescent_polarization
        self.crescent_intensity = crescent_intensity

        self.trigger_polarization = trigger_polarization
        self.trigger_intensity = trigger_intensity


class Experiment:
    """
    Just written for triplets at the moment.
    Provide a single value for each variable.
    """
    def __init__(self,
                 molecule, num_molecules, rotational_diffusion_time_ns,
                 collection_time_point_ns, repetitions,
                 excitation_scheme,
                 singlet_polarization, singlet_intensity,
                 crescent_polarization, crescent_intensity,
                 trigger_polarization, trigger_intensity,
                 ):
        # molecule properties
        self.molecule = molecule
        self.num_molecules = num_molecules
        self.rotational_diffusion_time_ns = rotational_diffusion_time_ns  # not pi adjusted

        # collection properties
        self.collection_time_point_ns = collection_time_point_ns
        self.repetitions = repetitions

        # excitation properties
        self.excitation_scheme = excitation_scheme
        self.singlet_polarization = singlet_polarization
        self.singlet_intensity = singlet_intensity
        self.crescent_polarization = crescent_polarization
        self.crescent_intensity = crescent_intensity
        self.trigger_polarization = trigger_polarization
        self.trigger_intensity = trigger_intensity

        # outputs
        self.num_x = 0
        self.num_y = 0
        self.ratio_xy_mean = None
        self.ratio_xy_std = None

        self.datetime = datetime.now().strftime("%Y%m%d%H%M%S")

    def _get_detector_counts(self, fluorophores, from_state, to_state, collection_times):
        x, y, _, t, = fluorophores.get_xyzt_at_transitions(from_state, to_state)
        t_collection = t[(t >= collection_times[0]) & (t <= collection_times[1])]
        x_collection = x[(t >= collection_times[0]) & (t <= collection_times[1])]
        y_collection = y[(t >= collection_times[0]) & (t <= collection_times[1])]
        t_x_temp, t_y_temp = utils.general.split_counts_xy(x_collection, y_collection, t_collection)
        self.num_x += len(t_x_temp)
        self.num_y += len(t_y_temp)
        if (len(t_x_temp) > 0) and (len(t_y_temp) > 0):
            ratio_xy = len(t_x_temp) / len(t_y_temp)
        elif len(t_x_temp) > 0:
            ratio_xy = 0
        else:
            ratio_xy = np.nan
        return ratio_xy

    def run_experiment(self):
        molecule_properties = MoleculeProperties(
            self.molecule, self.num_molecules, self.rotational_diffusion_time_ns,
        )
        excitation_properties = ExcitationProperties(
            self.singlet_polarization, self.singlet_intensity,
            self.crescent_polarization, self.crescent_intensity,
            self.trigger_polarization, self.trigger_intensity,
        )
        ratios = []
        for rep_num in range(self.repetitions):

            rep_percentage = round(rep_num / self.repetitions * 100)
            if rep_percentage % 25 == 0:
                logger.info(f'Experiment repetitions: {int((rep_num / self.repetitions) * 100)}%')

            collection_times = self.excitation_scheme(
                fluorophores=molecule_properties.fluorophores,
                collection_time_point=self.collection_time_point_ns,
                excitation_properties=excitation_properties,
            )
            ratios.append(
                self._get_detector_counts(
                    molecule_properties.fluorophores,
                    'singlet', 'ground',
                    collection_times,
                )
            )
        self.ratio_xy_mean = np.mean(np.array(ratios))
        self.ratio_xy_std = np.std(np.array(ratios))

    def csv_save(self, csv_path):
        # get the list of attributes
        attributes = self.__dict__
        attr_names = list(attributes.keys())
        attr_values = []
        for value in list(attributes.values()):
            attr_values.append(value)

        # Check if the file exists
        if not os.path.exists(csv_path):
            # Open the file for writing
            with open(csv_path, 'w', newline='') as csvfile:
                # Create a csv.writer object
                writer = csv.writer(csvfile, delimiter=',')
                # Write the header row
                writer.writerow(attr_names)

        # Open the file for appending
        with open(csv_path, 'a', newline='') as csvfile:
            # Create a csv.writer object
            writer = csv.writer(csvfile, delimiter=',')
            # Write each object to a row
            writer.writerow(attr_values)


if __name__ == "__main__":
    rotational_diffusion_times = range(150000, 250000, 1000)
    collection_times = range(1_000, 1500_000, 100_000)
    crescent_intensities = np.arange(0.1, 5, 0.1)
    csv_path = r'C:\Users\austin\GitHub\Rotational_diffusion-AELxJLD\rotational_diffusion\data\running_data.csv'
    for cres_num, crescent_intensity in enumerate(crescent_intensities):
        for rot_num, rotational_diffusion_time in enumerate(rotational_diffusion_times):
            experiments = []
            num_experiments = len(collection_times)
            for experiment_num, collection_time in enumerate(collection_times):
                logger.info(f'Experiment {experiment_num} of {len(collection_times)}, {rot_num} of {len(rotational_diffusion_times)}, {cres_num} of {len(crescent_intensities)}')
                experiment = Experiment(
                    molecule=variables.molecule_properties.mScarlet,
                    num_molecules=1E07,
                    rotational_diffusion_time_ns=rotational_diffusion_time * np.pi,
                    collection_time_point_ns=collection_time,
                    repetitions=100,
                    excitation_scheme=variables.excitation_schemes.pump_probe2,
                    singlet_polarization=(0, 1, 0), singlet_intensity=5,
                    crescent_polarization=(1, 0, 0), crescent_intensity=crescent_intensity,
                    trigger_polarization=(1, 0, 0), trigger_intensity=5,
                )
                experiment.run_experiment()
                experiment.csv_save(csv_path)
                experiments.append(experiment)

            means = []
            stds = []
            for experiment in experiments:
                means.append(experiment.ratio_xy_mean.get())  # todo cpu compat
                stds.append(experiment.ratio_xy_std.get())  # todo cpu compat
            plt.errorbar(range(num_experiments), means, yerr=stds)
    plt.show()
