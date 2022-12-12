import numpy as np

from rotational_diffusion.src.utils.general import sin_cos


class ElectronicState:
    """Information about an electronic state of a fluorophore."""
    def __init__(self, name: str, lifetime: float,
                 transition_states=None,
                 probabilities=None):
        self.name = name
        self.lifetime = lifetime

        # validate input
        assert isinstance(self.name, str)
        assert isinstance(self.lifetime, (int, float))
        assert self.lifetime > 0

        # set default and clean up transition states
        if transition_states is None:
            transition_states = [self.name]
        if isinstance(transition_states, str):
            transition_states = [transition_states]
        for idx, transition_state in enumerate(transition_states):
            if not isinstance(transition_state, str):
                raise ValueError("transition states must be a string or list of strings")
        self.transition_states = transition_states

        # set default and clean up probabilities
        if probabilities is None:
            if len(self.transition_states) == 1:
                probabilities = [1]
            else:
                raise ValueError("probabilities must be provided when multiple transition states are provided")
        if not isinstance(probabilities, list):
            probabilities = [probabilities]
        for probability in probabilities:
            assert isinstance(probability, (int, float))
        probabilities = np.asarray(probabilities, 'float64')
        assert probabilities.shape == (len(self.transition_states),)
        assert np.all(probabilities > 0)
        self.probabilities = probabilities / probabilities.sum()


class PossibleStates:
    """Holds a list of possible fluorophore electronic states."""
    def __init__(self):
        self.list = []
        self.dict = {}
        self.valid = True
        self.num = {}


class Orientations:
    """
    A class to simulate the orientations of an ensemble of freely
    rotating molecules, effectively a random walk on a sphere.

    Time evolution consists of rotational diffusion of orientation.
    You get to choose the "diffusion_time" (roughly, how long it
    takes the molecules to scramble their orientations).
    """
    def __init__(
            self,
            num_molecules,
            rot_diffusion_time,
            initial_orientations='uniform',
    ):
        assert num_molecules >= 1
        self.n = int(num_molecules)
        self.t = np.zeros(self.n, 'float64')

        rot_diffusion_time = np.asarray(rot_diffusion_time)
        assert rot_diffusion_time.shape in ((), (1,), (self.n,))
        assert np.all(rot_diffusion_time > 0)
        self.diffusion_time = rot_diffusion_time

        assert initial_orientations in ('uniform', 'polar')
        # Everybody starts at the north pole:
        self.x = np.zeros(self.n)
        self.y = np.zeros(self.n)
        self.z = np.ones(self.n)
        if initial_orientations == 'uniform':
            self.assign_uniform_positions()

    def assign_uniform_positions(self):
        # Generate random points on a sphere:
        sin_ph, cos_ph = sin_cos(np.random.uniform(0, 2 * np.pi, self.n), '0,2pi')
        cos_th = np.random.uniform(-1, 1, self.n)
        sin_th = np.sqrt(1 - cos_th * cos_th)
        self.x = sin_th * cos_ph
        self.y = sin_th * sin_ph
        self.z = cos_th


class Fluorophores:
    """
    Generates a number of fluorophores with specified diffusion times and fluorophore states based on the
    FluorophoreStateInfo object. Contains methods to manipulate the fluorophores, such as
    simulating photo transitions, rotational diffusion, and fluorophore deletions.
    """
    def __init__(
            self,
            num_molecules,
            rot_diffusion_time,
            state_info,
            orientation_initial='uniform',
            state_initial=0,
    ):
        self.orientations = Orientations(num_molecules, rot_diffusion_time, orientation_initial)



