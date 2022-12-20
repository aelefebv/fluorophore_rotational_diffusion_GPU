from rotational_diffusion.src.components import fluorophore


def create_triplet_state_info(molecule):
    ground_state = fluorophore.ElectronicState('ground')
    triplet_state = fluorophore.ElectronicState(
        'triplet', lifetime=molecule.triplet_lifetime_ns, transition_states='ground'
    )
    singlet_state = fluorophore.ElectronicState(
        'singlet', lifetime=molecule.singlet_lifetime_ns,
        transition_states=['ground', 'triplet'],
        probabilities=[1 - molecule.triplet_quantum_yield, molecule.triplet_quantum_yield]
    )
    state_info = fluorophore.PossibleStates(ground_state)
    state_info.add_state(triplet_state)
    state_info.add_state(singlet_state)
    return state_info


def create_singlet_state_info(molecule):
    ground_state = fluorophore.ElectronicState('ground')
    singlet_state = fluorophore.ElectronicState(
        'singlet', lifetime=molecule.singlet_lifetime_ns,
        transition_states='ground',
    )
    state_info = fluorophore.PossibleStates(ground_state)
    state_info.add_state(singlet_state)
    return state_info
