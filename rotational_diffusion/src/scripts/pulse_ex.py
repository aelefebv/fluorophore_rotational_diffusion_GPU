import rotational_diffusion.src.components.fluorophore
from rotational_diffusion.src import utils, experiments
import numpy as np
import matplotlib.pyplot as plt


my_fluorophore = experiments.molecule_properties.rsEGFP2()
number_of_molecules = 1E5
my_state_info = utils.state_info_creator.create_singlet_state_info(my_fluorophore)
f = rotational_diffusion.src.components.fluorophore.FluorophoreCollection(
    num_molecules=number_of_molecules,
    state_info=my_state_info,
    rot_diffusion_time=my_fluorophore.rotational_diffusion_time_ns
)
# pulse_schemes.starss_method1(f)
experiments.excitation_schemes.starss_method2(f)
x, y, z, t, = f.get_xyzt_at_transitions('singlet', 'ground')
xf, yf, tf = x[t > 5E05], y[t > 5E05], t[t > 5E05]
t_x, t_y = utils.general.split_counts_xy(xf, yf, tf)

ratiometric = utils.plotting_tools.plot_ratiometric_anisotropy(f, t_x, t_y, title='rsEGFP2_method2_anisotropy', save=False, x_space=(5, 6, 500))
utils.plotting_tools.plot_counts(f, t_x, t_y, title='rsEGFP2_method2_counts', save=False, x_space=(4E5, 7E5, 500))
