## from julia
import numpy as np

# Calculate the theoretical rotational correlation times for different bead sizes

kb = 1.380649e-23  # m^2 kg s^-2 K^-1
T_K = 298  # temperature in Kelvin
viscosity = 0.000890  # viscosity of water at 25C (298 K)


# viscosity units are m^-1 kg s^-1

def rot_corr_us_from_radius_m(radius):
    t_s = viscosity * (4 * np.pi * radius ** 3 / 3) / (kb * T_K)
    return (t_s * 1e6)


corr_us = []
for rad in [20e-9, 30e-9, 50e-9, 100e-9]:
    corr_us.append(rot_corr_us_from_radius_m(rad))

print(corr_us)

print(rot_corr_us_from_radius_m(25e-9))
