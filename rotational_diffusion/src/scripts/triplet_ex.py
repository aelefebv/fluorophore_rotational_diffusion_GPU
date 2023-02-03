from rotational_diffusion.src import utils, components, variables, np


my_fluorophore = variables.molecule_properties.mScarlet(100*np.pi)
number_of_molecules = 5E6
experiment = components.experiment.StateWriter(my_fluorophore, number_of_molecules, triplet=True)

pulse_len = 200
variables.excitation_schemes.standard_pulse(experiment.fluorophores, pulse_len)
experiment.fluorophores.delete_fluorophores_in_state('ground')

num_triplets = np.count_nonzero(experiment.fluorophores.states == 1)
print(f"[INFO] Triplets generated during pulsing: "
      f"{num_triplets} or {num_triplets/number_of_molecules*100:.2f}% of molecules")

variables.excitation_schemes.triplet_to_singlet(experiment.fluorophores)
x, y, z, t, = experiment.fluorophores.get_xyzt_at_transitions('singlet', 'ground')
xf, yf, tf = x[t > my_fluorophore.singlet_lifetime_ns], \
             y[t > my_fluorophore.singlet_lifetime_ns], \
             t[t > my_fluorophore.singlet_lifetime_ns]
t_x, t_y = utils.general.split_counts_xy(xf, yf, tf)

ratiometric = utils.plotting_tools.plot_ratiometric_anisotropy(experiment.fluorophores, t_x, t_y, title='mScarlet_anisotropy', save=False, x_space=(2, 6, 500))
utils.plotting_tools.plot_counts(experiment.fluorophores, t_x, t_y, title='mScarlet_counts', save=False, x_space=(pulse_len, 300, 500))

