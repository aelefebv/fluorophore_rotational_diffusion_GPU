from rotational_diffusion.src import utils, components, variables
import numpy as np
import matplotlib.pyplot as plt


class Sample:
    def __init__(self, fluorophore):
        self.fluorophore = fluorophore
        self.t_x = []
        self.t_y = []
        self.experiment = None

    def run_experiment(self, num_molecules, pulse_scheme, use_triplets=True):
        self.experiment = components.experiment.Experiment(self.fluorophore, num_molecules, triplet=use_triplets)
        pulse_scheme(self.experiment.fluorophores)

    def get_detector_counts(self, from_state, to_state):
        x, y, z, t, = self.experiment.fluorophores.get_xyzt_at_transitions(from_state, to_state)
        t_x_temp, t_y_temp = utils.general.split_counts_xy(x, y, t)
        self.t_x = np.concatenate([self.t_x, t_x_temp])
        self.t_y = np.concatenate([self.t_y, t_y_temp])


my_fluorophore_huge_bead = variables.molecule_properties.mScarlet(1000000 * np.pi)
my_fluorophore_100nm_bead = variables.molecule_properties.mScarlet(100000 * np.pi)
my_fluorophore_small = variables.molecule_properties.mScarlet(100 * np.pi)
sample_list = [my_fluorophore_small, my_fluorophore_100nm_bead, my_fluorophore_huge_bead]
num_molecules = 1E06
repetitions = 20
samples = []
for single_sample in sample_list:
    sample = Sample(single_sample)
    for rep_num in range(repetitions):
        print(f'[INFO] Experiment repetitions: {(rep_num / repetitions) * 100:.2f}%', end='\r')
        sample.run_experiment(num_molecules, variables.excitation_schemes.sp8_pulse_scheme)
        sample.get_detector_counts('singlet', 'ground')
    samples.append(sample)

fig, axes = plt.subplots(1, 2)

for single_sample in samples:
    utils.plotting_tools.plot_ratiometric_anisotropy(single_sample.experiment.fluorophores, single_sample.t_x, single_sample.t_y, title='mScarlet anisotropy from singlets', save=False, x_space=(0, 15, 100), log=False, plot_keep=axes[0])
    utils.plotting_tools.plot_ratiometric_anisotropy(single_sample.experiment.fluorophores, single_sample.t_x, single_sample.t_y, title='mScarlet anisotropy from triplets', save=False, x_space=(60195, 60195+single_sample.fluorophore.singlet_lifetime_ns*10, 100), log=False, plot_keep=axes[1])
    # utils.plotting_tools.plot_counts(sample.experiment.fluorophores, sample.t_x, sample.t_y, title='mScarlet counts from singlets', save=False, x_space=(-5, 220, 500))

plt.tight_layout()
plt.show()

fig2, axes2 = plt.subplots(2, 3)

for idx, single_sample in enumerate(samples):
    utils.plotting_tools.plot_counts(single_sample.experiment.fluorophores, single_sample.t_x, single_sample.t_y, title='mScarlet counts from singlets', save=False, x_space=(0, 15, 500), plot_keep=axes2[0, idx])
    utils.plotting_tools.plot_counts(single_sample.experiment.fluorophores, single_sample.t_x, single_sample.t_y, title='mScarlet counts from triplets', save=False, x_space=(60195, 60195+single_sample.fluorophore.singlet_lifetime_ns*10, 500), plot_keep=axes2[1, idx])
    # utils.plotting_tools.plot_counts(sample.experiment.fluorophores, sample.t_x, sample.t_y, title='mScarlet counts from singlets', save=False, x_space=(-5, 220, 500))

plt.tight_layout()
plt.show()



