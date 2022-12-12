from rotational_diffusion import fluorophore_rotational_diffusion


def create_triplet_state_info(fluorophore):
    state_info = fluorophore_rotational_diffusion.FluorophoreStateInfo()
    state_info.add('ground')
    state_info.add('triplet', lifetime=fluorophore.triplet_lifetime_ns, final_states='ground')
    # state_info.add('singlet', lifetime=fluorophore.singlet_lifetime_ns, final_states='ground')
    state_info.add(
        'singlet', lifetime=fluorophore.singlet_lifetime_ns,
        final_states=['ground', 'triplet'],
        probabilities=[1-fluorophore.triplet_quantum_yield, fluorophore.triplet_quantum_yield]
    )
    return state_info


def create_singlet_state_info(fluorophore):
    state_info = fluorophore_rotational_diffusion.FluorophoreStateInfo()
    state_info.add('ground')
    state_info.add(
        'singlet', lifetime=fluorophore.singlet_lifetime_ns,
        final_states='ground',
    )
    return state_info
