from rotational_diffusion.src import np


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


def triplet_to_singlet(fluorophores, capture_len_ns=5000, interval_ns=5, intensity=10, polarization_xyz=(1, 0, 0)):
    """Capture triplet emission."""
    reps = int(np.floor(capture_len_ns / interval_ns))  # don't do final capture if <1 full capture
    for _ in range(reps):
        fluorophores.time_evolve(interval_ns)
        fluorophores.delete_fluorophores_in_state('ground')
        fluorophores.phototransition('triplet', 'singlet', intensity=intensity, polarization_xyz=polarization_xyz)


def sp8_pulse_scheme(fluorophores, pulse_len_ns=200, pulse_freq_hz=80E6,
                     singlet_decay_len_ns=1E03, collection_interval_ns=60E03, triplet_interval_ns=2.5,
                     intensity_singlet=1, intensity_triplet=3,
                     polarization_xyz_singlet=(0, 1, 0), polarization_xyz_triplet=(1, 0, 0), intensity_photobleach=0):
    collection_start_time = pulse_len_ns+collection_interval_ns
    collection_end_time = collection_start_time + singlet_decay_len_ns

    rep_rate_ns = 1E9 * 1 / pulse_freq_hz
    reps = int(np.floor(pulse_len_ns / rep_rate_ns))  # don't do final pulse if <1 full pulse

    for _ in range(reps):
        # singlet "cap" excitation
        fluorophores.phototransition('ground', 'singlet',
                                     intensity=intensity_singlet, polarization_xyz=polarization_xyz_singlet)
        # sted delay
        fluorophores.time_evolve(triplet_interval_ns)
        # sted triplet "crescent" trigger
        if intensity_photobleach > 0:
            fluorophores.phototransition('triplet', 'bleached',
                                         intensity=intensity_photobleach, polarization_xyz=polarization_xyz_triplet)
            fluorophores.delete_fluorophores_in_state('bleached')
        fluorophores.phototransition('triplet', 'singlet',
                                     intensity=intensity_triplet, polarization_xyz=polarization_xyz_triplet)
        fluorophores.time_evolve(rep_rate_ns-triplet_interval_ns)

    fluorophores.delete_fluorophores_in_state('ground')
    fluorophores.time_evolve(collection_interval_ns)
    # remove any not in the crescent excitation
    fluorophores.delete_fluorophores_in_state('ground')
    # sted triplet collection trigger
    if intensity_photobleach > 0:
        fluorophores.phototransition('triplet', 'bleached',
                                     intensity=intensity_photobleach, polarization_xyz=polarization_xyz_triplet)
        fluorophores.delete_fluorophores_in_state('bleached')
    fluorophores.phototransition('triplet', 'singlet',
                                 intensity=intensity_triplet, polarization_xyz=polarization_xyz_triplet)
    fluorophores.time_evolve(singlet_decay_len_ns)
    fluorophores.delete_fluorophores_in_state('ground')

    return collection_start_time, collection_end_time

def cap_only(fluorophores, collection_interval_ns=10E03, singlet_decay_len_ns=1E03,
                   intensity_singlet=3, intensity_triplet=3,
                   polarization_xyz_singlet=(0, 1, 0), polarization_xyz_triplet=(1, 0, 0)):

    fluorophores.phototransition('ground', 'singlet',
                                 intensity=intensity_singlet, polarization_xyz=polarization_xyz_singlet)
    fluorophores.time_evolve(collection_interval_ns)
    fluorophores.phototransition('triplet', 'singlet',
                                 intensity=intensity_triplet, polarization_xyz=polarization_xyz_triplet)
    fluorophores.time_evolve(singlet_decay_len_ns)

    collection_start_time = collection_interval_ns
    collection_end_time = collection_start_time + singlet_decay_len_ns

    return collection_start_time, collection_end_time

def crescent_only(fluorophores, collection_interval_ns=10E03, triplet_transition_delay_ns=25, singlet_decay_len_ns=1E03,
                   intensity_singlet=3, intensity_crescent=3, intensity_triplet=3,
                   polarization_xyz_singlet=(0, 1, 0), polarization_xyz_triplet=(1, 0, 0)):

    fluorophores.phototransition('ground', 'singlet',
                                 intensity=intensity_singlet, polarization_xyz=polarization_xyz_singlet)
    fluorophores.time_evolve(triplet_transition_delay_ns)
    fluorophores.phototransition('triplet', 'singlet',
                                 intensity=intensity_crescent, polarization_xyz=polarization_xyz_triplet)
    fluorophores.time_evolve(collection_interval_ns)
    fluorophores.phototransition('triplet', 'singlet',
                                 intensity=intensity_triplet, polarization_xyz=polarization_xyz_triplet)
    fluorophores.time_evolve(singlet_decay_len_ns)

    collection_start_time = collection_interval_ns
    collection_end_time = collection_start_time + singlet_decay_len_ns

    return collection_start_time, collection_end_time


def pump_probe(fluorophores, collection_interval_ns=1E03, triplet_transition_delay_ns=25, singlet_decay_len_ns=1E03,
               intensity_singlet=5, intensity_crescent=0, intensity_triplet=5,
               polarization_xyz_singlet=(0, 1, 0),
               polarization_xyz_crescent=(1, 0, 0), polarization_xyz_triplet=(0, 0, 1)):
    fluorophores.phototransition('ground', 'singlet',
                                 intensity=intensity_singlet, polarization_xyz=polarization_xyz_singlet)
    fluorophores.time_evolve(triplet_transition_delay_ns)
    fluorophores.delete_fluorophores_in_state('ground')
    if intensity_crescent>0:
        fluorophores.phototransition('triplet', 'singlet',
                                     intensity=intensity_crescent, polarization_xyz=polarization_xyz_crescent)
    fluorophores.delete_fluorophores_in_state('ground')
    fluorophores.time_evolve(collection_interval_ns)
    fluorophores.delete_fluorophores_in_state('ground')
    fluorophores.phototransition('triplet', 'singlet',
                                 intensity=intensity_triplet, polarization_xyz=polarization_xyz_triplet)
    fluorophores.time_evolve(singlet_decay_len_ns)

    collection_start_time = collection_interval_ns
    collection_end_time = collection_start_time + singlet_decay_len_ns

    return collection_start_time, collection_end_time


def pump_probe2(fluorophores, collection_time_point, excitation_properties,
                triplet_transition_delay_ns=25, singlet_decay_len_ns=1E03):
    # excite molecules to ground state
    fluorophores.phototransition(
        'ground', 'singlet',
        intensity=excitation_properties.singlet_intensity,
        polarization_xyz=excitation_properties.singlet_polarization,
    )
    # let excited molecules go to ground or triplet
    fluorophores.time_evolve(triplet_transition_delay_ns)
    fluorophores.delete_fluorophores_in_state('ground')
    # crescent select if desired
    if excitation_properties.crescent_intensity > 0:
        fluorophores.phototransition(
            'triplet', 'singlet',
            intensity=excitation_properties.crescent_intensity,
            polarization_xyz=excitation_properties.crescent_polarization,
        )
        fluorophores.delete_fluorophores_in_state('ground')
    # beam comes back for triggering
    fluorophores.time_evolve(collection_time_point)
    fluorophores.delete_fluorophores_in_state('ground')
    # trigger triplets back to singlets
    fluorophores.phototransition(
        'triplet', 'singlet',
        intensity=excitation_properties.trigger_intensity,
        polarization_xyz=excitation_properties.trigger_polarization
    )
    # let singlets decay to ground
    fluorophores.time_evolve(singlet_decay_len_ns)
    # calculate collection window
    collection_start_time = collection_time_point
    collection_end_time = collection_start_time + singlet_decay_len_ns

    return collection_start_time, collection_end_time
