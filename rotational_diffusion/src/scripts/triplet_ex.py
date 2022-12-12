from rotational_diffusion import fluorophore_rotational_diffusion, pulse_schemes, fluorophores, utils
import numpy as np


def triplet_to_singlet(fluorophores, capture_len_ns=5000, interval_ns=5, intensity=10, polarization_xyz=(1, 0, 0)):
    """Capture triplet emission."""
    reps = int(np.floor(capture_len_ns / interval_ns))  # don't do final capture if <1 full capture
    for _ in range(reps):
        fluorophores.time_evolve(interval_ns)
        fluorophores.delete_fluorophores_in_state('ground')
        fluorophores.phototransition('triplet', 'singlet', intensity=intensity, polarization_xyz=polarization_xyz)


my_fluorophore = fluorophores.mScarlet()
number_of_molecules = 1E5
my_state_info = utils.state_info_creator.create_triplet_state_info(my_fluorophore)
f = fluorophore_rotational_diffusion.Fluorophores(
    number_of_molecules=number_of_molecules,
    state_info=my_state_info,
    diffusion_time=my_fluorophore.rotational_diffusion_time_ns
)
pulse_schemes.standard_pulse(f, 200)
f.delete_fluorophores_in_state('ground')
num_triplets = np.count_nonzero(f.states == 1)
print(f"[INFO] Triplets generated during pulsing: "
      f"{num_triplets} or {num_triplets/number_of_molecules*100:.2f}% of molecules")
triplet_to_singlet(f)
