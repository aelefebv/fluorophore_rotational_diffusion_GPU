from rotational_diffusion.src.components import fluorophore


def create_triplet_state_info(molecule, photobleach=False):
    ground_state = fluorophore.ElectronicState('ground')
    bleached_state = fluorophore.ElectronicState('bleached')
    if photobleach:
        triplet_transitions = ['ground', 'bleached']
        triplet_transition_probabilities = [0.9999, 0.0001]
    else:
        triplet_transitions = 'ground'
        triplet_transition_probabilities = 1
    triplet_state = fluorophore.ElectronicState(
        'triplet', lifetime=molecule.triplet_lifetime_ns,
        transition_states=triplet_transitions,
        probabilities=triplet_transition_probabilities
    )
    singlet_state = fluorophore.ElectronicState(
        'singlet', lifetime=molecule.singlet_lifetime_ns,
        transition_states=['ground', 'triplet'],
        probabilities=[1 - molecule.triplet_quantum_yield, molecule.triplet_quantum_yield]
    )
    state_info = fluorophore.PossibleStates(ground_state)
    state_info.add_state(triplet_state)
    state_info.add_state(singlet_state)
    if photobleach:
        state_info.add_state(bleached_state)
    return state_info


def create_singlet_state_info(molecule, photobleach=False):
    ground_state = fluorophore.ElectronicState('ground')
    singlet_state = fluorophore.ElectronicState(
        'singlet', lifetime=molecule.singlet_lifetime_ns,
        transition_states='ground',
    )
    state_info = fluorophore.PossibleStates(ground_state)
    state_info.add_state(singlet_state)
    return state_info
