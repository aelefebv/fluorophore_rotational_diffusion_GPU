from rotational_diffusion.src.components import fluorophore, experiment
from rotational_diffusion.src.utils import plotting_tools, state_info_creator, general
from rotational_diffusion.src.variables import molecule_properties, excitation_schemes


my_fluorophore = molecule_properties.mScarlet()
number_of_molecules = 1E5
test = experiment.StateWriter(my_fluorophore, number_of_molecules, triplet=True)
# pulse_schemes.starss_method1(f)
excitation_schemes.starss_method2(test.fluorophores)
x, y, z, t, = test.fluorophores.get_xyzt_at_transitions('singlet', 'ground')
xf, yf, tf = x[t > my_fluorophore.singlet_lifetime_ns], \
             y[t > my_fluorophore.singlet_lifetime_ns], \
             t[t > my_fluorophore.singlet_lifetime_ns]
t_x, t_y = general.split_counts_xy(xf, yf, tf)

ratiometric = plotting_tools.plot_ratiometric_anisotropy(test.fluorophores, t_x, t_y, title='rsEGFP2_method2_anisotropy', save=False, x_space=(5, 6, 500))
plotting_tools.plot_counts(test.fluorophores, t_x, t_y, title='rsEGFP2_method2_counts', save=False, x_space=(0, 300, 500))

