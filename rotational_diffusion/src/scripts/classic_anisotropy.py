from rotational_diffusion import fluorophore_rotational_diffusion
import matplotlib.pyplot as plt
from numpy.random import uniform
import numpy as np


### anisotropy example
n = int(1e7)
f = fluorophore_rotational_diffusion.Fluorophores(n)
print('Simulating', end='')
f.phototransition('ground', 'excited', intensity=0.05, polarization_xyz=(1, 0, 0))
f.phototransition('excited', 'triplet', intensity=0.05, polarization_xyz=(1, 0, 0))
while f.orientations.n > 0:
    print('.', end='')
    f.delete_fluorophores_in_state('ground')
    f.time_evolve(0.3)
print('done')
x, y, z, t = f.get_xyzt_at_transitions('excited', 'ground')
p_x, p_y = x**2, y**2
r = uniform(0, 1, size=len(t))
in_channel_x = (r < p_x)
in_channel_y = (p_x <= r) & (r < p_x + p_y)
t_x, t_y = t[in_channel_x], t[in_channel_y]
bins = np.linspace(0, 3, 200)
bin_centers = (bins[1:] + bins[:-1])/2
(hist_x, _), (hist_y, _) = np.histogram(t_x, bins), np.histogram(t_y, bins)

plt.figure()
plt.plot(bin_centers, hist_x)
plt.plot(bin_centers, hist_y)
plt.title(f"anisotropy\n"
          rf"$\tau_f$ = {f.state_info['excited'].lifetime}, "
          rf"$\tau_d$ = {f.orientations.diffusion_time}")
plt.xlabel = r"time (t/$\tau_f$)"
plt.ylabel = "photons per time bin"
# plt.legend()
plt.grid('on')
plt.show()
