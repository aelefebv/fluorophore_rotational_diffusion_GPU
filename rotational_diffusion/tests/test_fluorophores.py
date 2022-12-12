from rotational_diffusion.src.experiment.fluorophores import ElectronicState
import numpy as np
import pytest


def test_electronic_state_name():
    state = ElectronicState("S0", 1)
    assert state.name == "S0"


def test_electronic_state_lifetime():
    state = ElectronicState("S0", 1)
    assert state.lifetime == 1


def test_electronic_state_transition_states():
    state = ElectronicState("S0", 1, ["S1", "S2"], [0.5, 0.5])
    assert state.transition_states == ["S1", "S2"]


def test_electronic_state_probabilities():
    state = ElectronicState("S0", 1, ["S1", "S2"], [0.5, 0.5])
    assert np.allclose(state.probabilities, [0.5, 0.5])


def test_electronic_state_probabilities_sum_one():
    state = ElectronicState("S0", 1, ["S1", "S2"], [0.3, 0.2])
    assert state.probabilities.sum() == 1
    assert np.allclose(state.probabilities, [0.6, 0.4])


def test_electronic_state_transition_states_default():
    state = ElectronicState("S0", 1)
    assert state.transition_states == ["S0"]


def test_electronic_state_probabilities_default():
    state = ElectronicState("S0", 1)
    assert state.probabilities == [1]


def test_electronic_state_name_type():
    with pytest.raises(AssertionError):
        ElectronicState(1, 1)


def test_electronic_state_lifetime_type():
    with pytest.raises(AssertionError):
        ElectronicState("S0", "1")


def test_electronic_state_lifetime_positive():
    with pytest.raises(AssertionError):
        ElectronicState("S0", -1)


def test_electronic_state_transition_states_type():
    with pytest.raises(ValueError):
        ElectronicState("S0", 1, [1, 2])


def test_electronic_state_transition_states_probabilities_mismatch():
    with pytest.raises(AssertionError):
        ElectronicState("S0", 1, ["S1", "S2"], [0.5])


def test_electronic_state_probabilities_type():
    with pytest.raises(AssertionError):
        ElectronicState("S0", 1, ["S1", "S2"], ["0.5", "0.5"])
