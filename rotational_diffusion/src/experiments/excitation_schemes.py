import numpy as np


# 80MHz is pretty standard, assume infinitely narrow pulse width
def standard_pulse(fluorophores, pulse_len_ns=200, pulse_freq_hz=80E6, intensity=1, polarization_xyz=(0, 0, 1)):
    """Build up triplets."""
    rep_rate_ns = 1E9 * 1/pulse_freq_hz
    reps = int(np.floor(pulse_len_ns / rep_rate_ns))  # don't do final pulse if <1 full pulse
    for _ in range(reps):
        fluorophores.phototransition('ground', 'singlet', intensity=intensity, polarization_xyz=polarization_xyz)
        fluorophores.time_evolve(rep_rate_ns)


def standard_capture(fluorophores, capture_delay_ns=0.3):
    initial_n = fluorophores.orientations.n
    while fluorophores.orientations.n > 0:
        print(f'[INFO] Detecting emissions: {(1 - fluorophores.orientations.n/initial_n)*100:.2f}%', end='\r')
        fluorophores.delete_fluorophores_in_state('ground')
        fluorophores.time_evolve(capture_delay_ns)
    print()


def off_switching(fluorophores, pulse_freq_hz=80E6, intensity=1.0, polarization_xyz=(1, 0, 0)):
    initial_n = fluorophores.orientations.n
    rep_rate_ns = 1E9 * 1 / pulse_freq_hz
    while fluorophores.orientations.n > 0:
        print(f'[INFO] Detecting emissions: {(1 - fluorophores.orientations.n/initial_n)*100:.2f}%', end='\r')
        fluorophores.delete_fluorophores_in_state('ground')
        fluorophores.time_evolve(rep_rate_ns)
        fluorophores.phototransition('singlet', 'ground', intensity=intensity, polarization_xyz=polarization_xyz)


def off_switching_2(fluorophores, capture_delay_ns=100, intensity=1.0, polarization_xyz=(1, 0, 0)):
    initial_n = fluorophores.orientations.n
    while fluorophores.orientations.n > 0:
        print(f'[INFO] Detecting emissions: {(1 - fluorophores.orientations.n/initial_n)*100:.2f}%', end='\r')
        fluorophores.delete_fluorophores_in_state('ground')
        fluorophores.time_evolve(capture_delay_ns)
        fluorophores.phototransition('singlet', 'ground', intensity=intensity, polarization_xyz=polarization_xyz)


def starss_method1(fluorophores, capture_delay_ns=100):
    # preliminary off-pulse not included
    standard_pulse(fluorophores, pulse_len_ns=250, polarization_xyz=(1, 0, 0))  # polarized on
    standard_capture(fluorophores, capture_delay_ns)  # read out


def starss_method2(fluorophores):
    # preliminary off-pulse not included
    standard_pulse(fluorophores, pulse_len_ns=250, polarization_xyz=(0, 0, 1))  # unpolarized on
    fluorophores.delete_fluorophores_in_state('ground')
    fluorophores.time_evolve(5E05)  # wait time of 500 us
    off_switching(fluorophores, intensity=0.01)



