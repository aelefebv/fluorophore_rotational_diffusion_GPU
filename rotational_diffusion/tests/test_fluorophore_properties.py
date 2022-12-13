from rotational_diffusion.src.components.fluorophore import ElectronicState, PossibleStates
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
    with pytest.raises(TypeError):
        ElectronicState("S0", 1, [1, 2])


def test_electronic_state_transition_states_probabilities_mismatch():
    with pytest.raises(AssertionError):
        ElectronicState("S0", 1, ["S1", "S2"], [0.5])


def test_electronic_state_probabilities_type():
    with pytest.raises(AssertionError):
        ElectronicState("S0", 1, ["S1", "S2"], ["0.5", "0.5"])

def test_possible_states_init_no_transition():
    # Create an ElectronicState object
    electronic_state = ElectronicState(name="S0", lifetime=10)

    # Instantiate a PossibleStates object with the ElectronicState object as the electronic_state parameter
    possible_states = PossibleStates(electronic_state=electronic_state)

    # Verify that the PossibleStates.dict attribute is initialized from the electronic state
    assert possible_states.dict == {"S0": electronic_state}

    # Verify that the PossibleStates.valid attribute is initialized as True if no final state is given
    assert possible_states.valid

    # Verify that the PossibleStates.orphan_states attribute is initialized as an empty list
    assert possible_states.orphan_states == []

    # Verify that the ElectronicState.state_num attribute of the ElectronicState object is set to 0
    assert electronic_state.state_num == 0

    # Verify that the ElectronicState object  is added to the PossibleStates.dict dictionary
    assert "S0" in possible_states.dict


def test_possible_states_init_with_transition():
    # Create an ElectronicState object
    electronic_state = ElectronicState(name="S0", lifetime=10, transition_states="S1")

    # Instantiate a PossibleStates object with the ElectronicState object as the electronic_state parameter
    possible_states = PossibleStates(electronic_state=electronic_state)

    # Verify that the PossibleStates.dict attribute is initialized from the electronic state
    assert possible_states.dict == {"S0": electronic_state}

    # Verify that the PossibleStates.valid attribute is initialized as False if a final state exists but is not added
    assert not possible_states.valid

    # Verify that the PossibleStates.orphan_states attribute contains the orphaned final state
    assert possible_states.orphan_states == [("S0", "S1")]

    # Verify that the ElectronicState.state_num attribute of the ElectronicState object is set to 0
    assert electronic_state.state_num == 0

    # Verify that the ElectronicState object  is added to the PossibleStates.dict dictionary
    assert "S0" in possible_states.dict


def test_add_single_state():
    # create a single state
    state = ElectronicState('S0', lifetime=1.0)
    possible_states = PossibleStates(state)

    # check that the state was added correctly
    assert possible_states.dict == {'S0': state}
    assert possible_states.valid
    assert possible_states.orphan_states == []


def test_add_invalid_state():
    # create a single state
    state1 = ElectronicState('S0', lifetime=1.0)
    possible_states = PossibleStates(state1)

    # create a second state with an invalid transition state
    state2 = ElectronicState('S1', lifetime=1.0, transition_states='S2')
    possible_states.add_state(state2)

    # check that the state was added correctly
    assert possible_states.dict == {'S0': state1, 'S1': state2}
    assert not possible_states.valid
    assert possible_states.orphan_states == [('S1', 'S2')]


def test_add_valid_state():
    # create a single state
    state1 = ElectronicState('S0', lifetime=1.0)
    possible_states = PossibleStates(state1)

    # create a second state with a valid transition state
    state2 = ElectronicState('S1', lifetime=1.0, transition_states='S0')
    possible_states.add_state(state2)

    # check that the state was added correctly
    assert possible_states.dict == {'S0': state1, 'S1': state2}
    assert possible_states.valid
    assert possible_states.orphan_states == []


def test_get_state_num_and_lifetime():
    # create a single state
    state1 = ElectronicState('S0', lifetime=1.0)
    possible_states = PossibleStates(state1)

    # create a second state with a valid transition state
    state2 = ElectronicState('S1', lifetime=2.0, transition_states='S0')
    possible_states.add_state(state2)

    # check that the state_num and lifetime of 'S0' can be retrieved correctly
    state_nums, lifetimes = possible_states.get_state_num_and_lifetime('S0')
    assert state_nums == [0]
    assert lifetimes == [1.0]

    # check that the state_num and lifetime of 'S1' can be retrieved correctly
    state_nums, lifetimes = possible_states.get_state_num_and_lifetime('S1')
    assert state_nums == [1]
    assert lifetimes == [2.0]


def test_get_state_num_and_lifetime_with_list():
    # create a single state
    state1 = ElectronicState('S0', lifetime=1.0)
    possible_states = PossibleStates(state1)

    # create a second state with a valid transition state
    state2 = ElectronicState('S1', lifetime=2.0, transition_states='S0')
    possible_states.add_state(state2)

    # check that the state_num and lifetime of ['S0', 'S1'] can be retrieved correctly
    state_nums, lifetimes = possible_states.get_state_num_and_lifetime(['S0', 'S1'])
    assert np.allclose(state_nums, [0, 1])
    assert np.allclose(lifetimes, [1.0, 2.0])


def test__validate():
    # Test that _validate correctly sets valid to True when all transition states exist in the dict.
    state1 = ElectronicState('S1', 1, ['S1'], [1])
    state2 = ElectronicState('S2', 2, ['S1'], [1])
    state3 = ElectronicState('S3', 3, ['S1', 'S2'], [1, 1])
    possible_states = PossibleStates(state1)
    possible_states.add_state(state2)
    possible_states.add_state(state3)
    assert possible_states.valid
    assert possible_states.orphan_states == []

    # Test that _validate correctly sets valid to False when one or more transition states do not exist in the dict.
    state4 = ElectronicState('S4', 4, ['S5'], [1])
    possible_states.add_state(state4)
    assert not possible_states.valid

    # Test that _validate correctly populates the orphan_states list with tuples of state names and unmatched
    # transition states when valid is False.
    assert possible_states.orphan_states == [('S4', 'S5')]


# def test_getitem():
#     # create some electronic states to use in the tests
#     s1 = ElectronicState("S1", 1, ["S1"])
#     s2 = ElectronicState("S2", 2, ["S2"])
#     s3 = ElectronicState("S3", 3, ["S1"])
#     s4 = ElectronicState("S4", 4, ["S2"])
#
#     # create a PossibleStates instance
#     ps = PossibleStates(s1)
#     ps.add(s2)
#     ps.add(s3)
#     ps.add(s4)
#
#     # test that IndexError is raised when __getitem__ is called
#     # with an item that is not in the dict attribute of the PossibleStates instance
#     with pytest.assertRaises(IndexError):
#         ps["S5"]
#
#     # test that the correct ElectronicState instance is returned when
#     # __getitem__ is called with a valid state name
#     pytest.assertEqual(ps["S1"].name, "S1")
#     pytest.assertEqual(ps["S1"].lifetime, 1)
#
#     # test that TypeError is raised when __get

def test_getitem():
    # create some ElectronicState objects
    e1 = ElectronicState("e1", 1, ["e1"], [1])
    e2 = ElectronicState("e2", 2, ["e2"], [1])
    e3 = ElectronicState("e3", 3, ["e1"], [1])
    e4 = ElectronicState("e4", 4, ["e5"], [1])

    # create a PossibleStates object and add the ElectronicState objects
    states = PossibleStates(e1)
    states.add_state(e2)
    states.add_state(e3)

    # test that the __getitem__ method returns the correct ElectronicState
    # object when called with a string or integer representing the name or
    # state number of a valid electronic state
    assert states["e1"] == e1
    assert states[0] == e1
    assert states["e2"] == e2
    assert states[1] == e2
    assert states["e3"] == e3
    assert states[2] == e3

    # test that the __getitem__ method raises a TypeError when called with
    # an invalid argument type (e.g. a list of strings and integers)
    with pytest.raises(TypeError):
        _ = states[["e1", 1]]

    states.add_state(e4)

    # test that the __getitem__ method raises a LookupError when called on an
    # object that is not valid (i.e., has unmatched final states)
    with pytest.raises(LookupError):
        _ = states["e4"]
