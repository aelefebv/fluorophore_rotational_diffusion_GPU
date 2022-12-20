from rotational_diffusion.src.components import fluorophore
import numpy as np


def test_fluorophores_initialization():
    num_molecules = 10
    rot_diffusion_time = 0.1
    state1 = fluorophore.ElectronicState('S1', 1, ['S1'], [1])
    state2 = fluorophore.ElectronicState('S2', 2, ['S1'], [1])
    state_info = fluorophore.PossibleStates(state1)
    state_info.add_state(state2)
    orientation_initial = 'uniform'
    state_initial = 0

    fluorophores = fluorophore.FluorophoreCollection(
        num_molecules,
        rot_diffusion_time,
        state_info,
        orientation_initial=orientation_initial,
        state_initial=state_initial,
    )

    assert isinstance(fluorophores.state_info, fluorophore.PossibleStates)
    assert state_initial in fluorophores.state_info
    assert isinstance(fluorophores.orientations, fluorophore.Orientations)
    assert isinstance(fluorophores.states, np.ndarray)
    assert fluorophores.states.dtype == 'uint8'
    assert len(fluorophores.states) == num_molecules
    assert (fluorophores.states == state_initial).all()
    assert isinstance(fluorophores.transition_times, np.ndarray)
    assert len(fluorophores.transition_times) == num_molecules
    assert isinstance(fluorophores.id, np.ndarray)
    assert len(fluorophores.id) == num_molecules
    assert isinstance(fluorophores.transition_events, dict)
    assert set(fluorophores.transition_events.keys()) == {'x', 'y', 'z', 't', 'initial_state', 'final_state'}
