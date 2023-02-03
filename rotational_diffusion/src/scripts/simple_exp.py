from rotational_diffusion.src import utils, components, variables, np
import numpy
import matplotlib.pyplot as plt

crescent_intensity = [0, 3]

my_fluorophore_medium = variables.molecule_properties.mScarlet(250000 * np.pi)
collection_intervals = range(1_000, 1500_000, 200_000)
sample_list = [my_fluorophore_medium]
num_molecules = 1E07
repetitions = 20
samples = []
for interval_num, collection_interval in enumerate(collection_intervals):
    print(f'[INFO] Collection interval {interval_num} of {len(collection_intervals)}')
    for single_sample in sample_list:
        sample = components.sample.Sample(single_sample)
        for rep_num in range(repetitions):
            print(f'[INFO] Experiment repetitions: {(rep_num / repetitions) * 100:.2f}%', end='\r')
            sample.experiment = components.experiment.Experiment(sample.fluorophore, num_molecules,
                                                                 triplet=True)
            collection_times = variables.excitation_schemes.pump_probe(
                sample.experiment.fluorophores,
                collection_interval_ns=collection_interval,
                # polarization_xyz_triplet=(1, 0, 0)
            )
            sample.get_detector_counts('singlet', 'ground', collection_times)
        samples.append(sample)

mean_triplet_ani = []
std_triplet_ani = []
for single_sample in samples:
    if len(single_sample.t_y):
        # mean_triplet_ani.append(len(single_sample.t_x)/len(single_sample.t_y))
        mean_triplet_ani.append(numpy.nanmean(numpy.array(single_sample.ratio_mean)))
        std_triplet_ani.append(numpy.nanstd(numpy.array(single_sample.ratio_mean)))
    else:
        mean_triplet_ani.append(numpy.nan)
        std_triplet_ani.append(numpy.nan)

# plt.figure()
plt.errorbar(collection_intervals, mean_triplet_ani, yerr=std_triplet_ani)

plt.show()
# samples_to_plot = [samples[0], samples[len(samples)//2], samples[-1]]
#
# fig, axes = plt.subplots(1, 1)
#
# aniso_triplets = []
# aniso_singlets = []
# for idx, single_sample in enumerate(samples_to_plot):
#     # _, aniso_singlet = utils.plotting_tools.plot_ratiometric_anisotropy(single_sample.experiment.fluorophores, single_sample.t_x, single_sample.t_y, title='mScarlet anisotropy from singlets', save=False, x_space=(0, 5, 100), log=False, plot_keep=axes[0])
#     # _, aniso_triplet = utils.plotting_tools.plot_ratiometric_anisotropy(single_sample.experiment.fluorophores, single_sample.t_x, single_sample.t_y, sub=collection_intervals[idx], title='mScarlet anisotropy from triplets', save=False, x_space=(60195, 60195+single_sample.fluorophore.singlet_lifetime_ns*10, 100), log=False, plot_keep=axes[1])
#     utils.plotting_tools.plot_ratiometric_anisotropy(single_sample.experiment.fluorophores, single_sample.t_x, single_sample.t_y, title='mScarlet anisotropy from triplets', sub=np.min(single_sample.t_x), save=False, x_space=(np.min(single_sample.t_x), np.max(single_sample.t_x), 50), log=False, plot_keep=axes)
#
#     # utils.plotting_tools.plot_counts(sample.experiment.fluorophores, sample.t_x, sample.t_y, title='mScarlet counts from singlets', save=False, x_space=(-5, 220, 500))
#     # aniso_singlets.append(aniso_singlet)
#
# plt.tight_layout()
# plt.show()
#
#
# fig2, axes2 = plt.subplots(1, len(samples_to_plot))
#
# for idx, single_sample in enumerate(samples_to_plot):
#     # utils.plotting_tools.plot_counts(single_sample.experiment.fluorophores, single_sample.t_x, single_sample.t_y, title='mScarlet counts from singlets', save=False, x_space=(0, 15, 500), plot_keep=axes2[0, idx])
#     utils.plotting_tools.plot_counts(single_sample.experiment.fluorophores, single_sample.t_x, single_sample.t_y, title='mScarlet counts from triplets', save=False, x_space=(np.min(single_sample.t_x), np.max(single_sample.t_x), 50), plot_keep=axes2[idx])
#     # utils.plotting_tools.plot_counts(sample.experiment.fluorophores, sample.t_x, sample.t_y, title='mScarlet counts from singlets', save=False, x_space=(-5, 220, 500))
#
# plt.tight_layout()
# plt.show()