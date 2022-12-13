from rotational_diffusion.src.utils import diffusive_steps
import numpy as np
import pytest


def test_ghosh_propagator_negative_input():
    # Test handling of negative input
    step_sizes = np.array([-1, 0, 1])
    with pytest.raises(AssertionError):
        diffusive_steps.ghosh_propagator(step_sizes)


def test_ghosh_propagator_0_value_input():
    # Test handling of 0-valued input
    step_sizes = np.array([0, 0, 0])
    result = diffusive_steps.ghosh_propagator(step_sizes)
    assert result.shape == step_sizes.shape
    assert np.all(result > 0)


def test_ghosh_propagator_positive_input():
    # Test output for positive input
    step_sizes = np.array([1, 2, 3])
    result = diffusive_steps.ghosh_propagator(step_sizes)
    assert result.shape == step_sizes.shape
    for r in result:
        assert 0 < r <= np.pi
