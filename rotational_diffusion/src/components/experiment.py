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
        self.rotational_diffusion_time_unpied = self.rotational_diffusion_time / np.pi
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
                 trigger_polarization, trigger_intensity,
                 num_triggers):

        self.singlet_polarization = singlet_polarization
        self.singlet_intensity = singlet_intensity

        self.crescent_polarization = crescent_polarization
        self.crescent_intensity = crescent_intensity

        self.trigger_polarization = trigger_polarization
        self.trigger_intensity = trigger_intensity

        self.num_triggers = num_triggers


class Experiment:
    """
    Just written for triplets at the moment.
    Provide a single value for each variable.
    """
    def __init__(self,
                 molecule_properties: MoleculeProperties,
                 excitation_properties: ExcitationProperties,
                 collection_time_point_ns,
                 repetitions,
                 excitation_scheme,
                 trigger_number,
                 collection_start_time
                 ):
        # molecule properties
        self.molecule = molecule_properties.molecule
        self.num_molecules = molecule_properties.num_molecules
        self.rotational_diffusion_time_ns = molecule_properties.rotational_diffusion_time  # not pi adjusted
        self.rotational_diffusion_time_ns_unpied = molecule_properties.rotational_diffusion_time / np.pi  # pi adjusted

        # collection properties
        self.collection_time_point_ns = collection_time_point_ns
        self.collection_time_point_us = collection_time_point_ns / 1000
        self.repetitions = repetitions

        self.excitation_scheme = excitation_scheme
        # excitation properties
        self.singlet_polarization = excitation_properties.singlet_polarization
        self.singlet_intensity = excitation_properties.singlet_intensity
        self.crescent_polarization = excitation_properties.crescent_polarization
        self.crescent_intensity = excitation_properties.crescent_intensity
        self.trigger_polarization = excitation_properties.trigger_polarization
        self.trigger_intensity = excitation_properties.trigger_intensity

        # outputs
        self.num_x = 0
        self.num_y = 0
        self.ratio_xy_mean = None
        self.ratio_xy_std = None

        self.datetime = datetime.now().strftime("%Y%m%d%H%M%S")

        self.trigger_number = trigger_number
        self.collection_start_time = collection_start_time

    def get_detector_counts(self, fluorophores, from_state, to_state, collection_times):
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

    def csv_save(self, csv_path):
        # get the list of attributes
        attributes = self.__dict__
        attr_names = list(attributes.keys())
        attr_values = []
        for value in list(attributes.values()):
            attr_values.append(value)

        # check if the directory exists
        if not os.path.exists(os.path.dirname(csv_path)):
            # make the directory
            os.makedirs(os.path.dirname(csv_path))

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


def run_experiment(molecule_props, excitation_props,
                   collection_time_point_ns,
                   repetitions,
                   excitation_scheme,
                   ):
    experiments = {}
    ratios = {}
    for rep_num in range(repetitions):

        rep_percentage = round(rep_num / repetitions * 100)
        if rep_percentage % 25 == 0:
            logger.info(f'Experiment repetitions: {int((rep_num / repetitions) * 100)}%')

        collection_times = excitation_scheme(
            fluorophores=molecule_props.fluorophores,
            collection_time_point=collection_time_point_ns,
            excitation_properties=excitation_props,
        )
        if not isinstance(collection_times, list):
            collection_times = [collection_times]
        for collection_time in collection_times:
            if collection_time[0] not in experiments:
                experiments[collection_time[0]] = Experiment(molecule_properties=molecule_props,
                                                             excitation_properties=excitation_props,
                                                             collection_time_point_ns=collection_time_point_ns,
                                                             repetitions=repetitions,
                                                             excitation_scheme=excitation_scheme,
                                                             trigger_number=collection_time[0],
                                                             collection_start_time=collection_time[1])
            if collection_time[0] not in ratios:
                ratios[collection_time[0]] = []
            ratios[collection_time[0]].append(
                experiments[collection_time[0]].get_detector_counts(
                    molecule_props.fluorophores,
                    'singlet', 'ground',
                    collection_time[1:],
                )
            )
    for trigger_number in ratios:
        experiments[trigger_number].ratio_xy_mean = np.nanmean(np.array(ratios[trigger_number]))
        experiments[trigger_number].ratio_xy_std = np.nanstd(np.array(ratios[trigger_number]))
    return experiments


def run_multi_variable():
    # Note that diffusion times are always multiplied by pi!
    # rotational_diffusion_times = [2500, 25000, 250000]  # for EGFR
    rotational_diffusion_times = [4630, 15640, 72400, 579200]  # for beads
    # rotational_diffusion_times = [2500, 250000]
    # rotational_diffusion_times = np.logspace(4, 8, num=20).tolist()
    # rotational_diffusion_times = np.logspace(4, 8, num=50)
    # rotational_diffusion_times = range(10_000, 100_000_000, 1000)
    # collection_times_exp = [400000, 1500000]
    collection_times_exp = np.linspace(1, 1E6, num=100).tolist()
    # collection_times_exp = np.logspace(0, 4, num=100).tolist()
    # collection_times_exp = np.logspace(3, 5, num=20)
    # collection_times_exp = range(1_000, 100_000, 5_000)
    crescent_intensities = [2, 4, 8]
    # crescent_intensities = np.logspace(-3, 2, num=10, base=2).tolist()
    singlet_intensities = [2]
    # singlet_intensities = [0.1, 0.5, 1, 2, 3]
    csv_path = r'C:\Users\austin\GitHub\Rotational_diffusion-AELxJLD\rotational_diffusion\data\20230411_beads.csv'
    # csv_path = r'C:\Users\austin\GitHub\Rotational_diffusion-AELxJLD\rotational_diffusion\data\running_data.csv'
    for sing_num, singlet_intensity in enumerate(singlet_intensities):
        for cres_num, crescent_intensity in enumerate(crescent_intensities):
            for rot_num, rotational_diffusion_time in enumerate(rotational_diffusion_times):
                for experiment_num, collection_time_exp in enumerate(collection_times_exp):
                    collection_time_exp = int(collection_time_exp)
                    logger.info(f'Experiment {experiment_num} of {len(collection_times_exp)}, '
                                f'{rot_num} of {len(rotational_diffusion_times)}, '
                                f'{cres_num} of {len(crescent_intensities)}, '
                                f'{sing_num} of {len(singlet_intensities)}')
                    molecule_properties = MoleculeProperties(
                        molecule=variables.molecule_properties.mScarlet,
                        num_molecules=2E07,
                        rotational_diffusion_time=rotational_diffusion_time * np.pi,
                    )
                    excitation_properties = ExcitationProperties(
                        singlet_polarization=(0, 1, 0), singlet_intensity=singlet_intensity,
                        crescent_polarization=(1, 0, 0), crescent_intensity=crescent_intensity,
                        trigger_polarization=(1, 0, 0), trigger_intensity=0.25,
                        num_triggers=1,
                    )
                    experiments = run_experiment(
                        molecule_properties, excitation_properties,
                        collection_time_point_ns=collection_time_exp,
                        repetitions=50,
                        excitation_scheme=variables.excitation_schemes.gentle_check,
                    )
                    for experiment in experiments:
                        experiments[experiment].csv_save(csv_path)

                # experiments.append(experiment)
    #         means = []
    #         stds = []
    #         for experiment in experiments:
    #             means.append(experiment.ratio_xy_mean.get())  # todo cpu compat
    #             stds.append(experiment.ratio_xy_std.get())  # todo cpu compat
    #         plt.errorbar(range(num_experiments), means, yerr=stds)
    # plt.show()


if __name__ == "__main__":
    run_multi_variable()
    # molecule_properties = MoleculeProperties(
    #     molecule=variables.molecule_properties.mScarlet,
    #     num_molecules=2E07,
    #     rotational_diffusion_time=100000 * np.pi,
    # )
    # excitation_properties = ExcitationProperties(
    #     singlet_polarization=(0, 1, 0), singlet_intensity=3,
    #     crescent_polarization=(1, 0, 0), crescent_intensity=0,
    #     trigger_polarization=(1, 0, 0), trigger_intensity=3,
    #     num_triggers=30,
    # )
    # experiments = run_experiment(
    #     molecule_properties, excitation_properties,
    #     collection_time_point_ns=collection_time_exp,
    #     repetitions=50,
    #     excitation_scheme=variables.excitation_schemes.pump_probe2,
    # )
