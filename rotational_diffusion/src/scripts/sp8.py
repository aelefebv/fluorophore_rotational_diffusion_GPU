from rotational_diffusion.src import utils, components, variables, np
import matplotlib.pyplot as plt


class Sample:
    def __init__(self, fluorophore):
        self.fluorophore = fluorophore
        self.t_x = []
        self.t_y = []
        self.experiment = None

    def run_experiment(self, num_molecules, pulse_scheme, use_triplets=True):
        self.experiment = components.experiment.StateWriter(self.fluorophore, num_molecules, triplet=use_triplets)
        pulse_scheme(self.experiment.fluorophores)

    def get_detector_counts(self, from_state, to_state, collection_times):
        x, y, _, t, = self.experiment.fluorophores.get_xyzt_at_transitions(from_state, to_state)
        t_collection = t[(t >= collection_times[0]) & (t <= collection_times[1])]
        x_collection = x[(t >= collection_times[0]) & (t <= collection_times[1])]
        y_collection = y[(t >= collection_times[0]) & (t <= collection_times[1])]
        t_x_temp, t_y_temp = utils.general.split_counts_xy(x_collection, y_collection, t_collection)
        self.t_x = np.concatenate([self.t_x, t_x_temp])
        self.t_y = np.concatenate([self.t_y, t_y_temp])

# my_fluorophore_huge_bead = variables.molecule_properties.mScarlet(1000000 * np.pi)
# my_fluorophore_superbig_bead = variables.molecule_properties.mScarlet(100000000 * np.pi)
my_fluorophore_100nm_bead = variables.molecule_properties.mScarlet(100000 * np.pi)
# my_fluorophore_50nm_bead = variables.molecule_properties.mScarlet(10000 * np.pi)
# my_fluorophore_100nm_bead_long = variables.molecule_properties.mScarlet_high_triplet(100000 * np.pi)
# my_fluorophore_100nm_bead_short = variables.molecule_properties.mScarlet_low_triplet(100000 * np.pi)
# my_fluorophore_small = variables.molecule_properties.mScarlet(100 * np.pi)
# sample_list = [my_fluorophore_small, my_fluorophore_100nm_bead, my_fluorophore_huge_bead]
sample_list = [my_fluorophore_100nm_bead]
# collection_intervals = [60E03, 120E03, 180E03, 240E03, 300E03, 1000E03]
collection_intervals = range(60_000, 1000_000, 200_000)
# collection_intervals = [1000E03]

num_molecules = 5E04
repetitions = 10
samples = []
for interval_num, collection_interval in enumerate(collection_intervals):
    print(f'[INFO] Collection interval {interval_num} of {len(collection_intervals)}')
    for single_sample in sample_list:
        sample = Sample(single_sample)
        for rep_num in range(repetitions):
            print(f'[INFO] Experiment repetitions: {(rep_num / repetitions) * 100:.2f}%', end='\r')
            sample.experiment = components.experiment.StateWriter(sample.fluorophore, num_molecules,
                                                                  triplet=True, photobleach=True)
            collection_times = variables.excitation_schemes.sp8_pulse_scheme(
                sample.experiment.fluorophores,
                collection_interval_ns=collection_interval)
            sample.get_detector_counts('singlet', 'ground', collection_times)
        samples.append(sample)

mean_triplet_ani = []
for single_sample in samples:
    mean_triplet_ani.append(len(single_sample.t_x)/len(single_sample.t_y))

plt.figure()
plt.plot(collection_intervals, mean_triplet_ani)
plt.show()

samples_to_plot = [samples[0], samples[len(samples)//2], samples[-1]]

fig, axes = plt.subplots(1, 1)

aniso_triplets = []
aniso_singlets = []
for idx, single_sample in enumerate(samples_to_plot):
    # _, aniso_singlet = utils.plotting_tools.plot_ratiometric_anisotropy(single_sample.experiment.fluorophores, single_sample.t_x, single_sample.t_y, title='mScarlet anisotropy from singlets', save=False, x_space=(0, 5, 100), log=False, plot_keep=axes[0])
    # _, aniso_triplet = utils.plotting_tools.plot_ratiometric_anisotropy(single_sample.experiment.fluorophores, single_sample.t_x, single_sample.t_y, sub=collection_intervals[idx], title='mScarlet anisotropy from triplets', save=False, x_space=(60195, 60195+single_sample.fluorophore.singlet_lifetime_ns*10, 100), log=False, plot_keep=axes[1])
    utils.plotting_tools.plot_ratiometric_anisotropy(single_sample.experiment.fluorophores, single_sample.t_x, single_sample.t_y, title='mScarlet anisotropy from triplets', sub=np.min(single_sample.t_x), save=False, x_space=(np.min(single_sample.t_x), np.max(single_sample.t_x), 50), log=False, plot_keep=axes)

    # utils.plotting_tools.plot_counts(sample.experiment.fluorophores, sample.t_x, sample.t_y, title='mScarlet counts from singlets', save=False, x_space=(-5, 220, 500))
    # aniso_singlets.append(aniso_singlet)

plt.tight_layout()
plt.show()


fig2, axes2 = plt.subplots(1, len(samples_to_plot))

for idx, single_sample in enumerate(samples_to_plot):
    # utils.plotting_tools.plot_counts(single_sample.experiment.fluorophores, single_sample.t_x, single_sample.t_y, title='mScarlet counts from singlets', save=False, x_space=(0, 15, 500), plot_keep=axes2[0, idx])
    utils.plotting_tools.plot_counts(single_sample.experiment.fluorophores, single_sample.t_x, single_sample.t_y, title='mScarlet counts from triplets', save=False, x_space=(np.min(single_sample.t_x), np.max(single_sample.t_x), 50), plot_keep=axes2[idx])
    # utils.plotting_tools.plot_counts(sample.experiment.fluorophores, sample.t_x, sample.t_y, title='mScarlet counts from singlets', save=False, x_space=(-5, 220, 500))

plt.tight_layout()
plt.show()
