import numpy as np

from rotational_diffusion.src import utils


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
        self.n = int(num_molecules)
        self.t = np.zeros(self.n, 'float64')
        rot_diffusion_time = np.asarray(rot_diffusion_time)
        self.rot_diffusion_time = rot_diffusion_time

        assert num_molecules >= 1
        assert rot_diffusion_time.shape in ((), (1,), (self.n,))
        assert np.all(rot_diffusion_time > 0)
        assert initial_orientations in ('uniform', 'polar')

        # Everybody starts at the north pole:
        self.x = np.zeros(self.n)
        self.y = np.zeros(self.n)
        self.z = np.ones(self.n)
        if initial_orientations == 'uniform':
            self._assign_uniform_positions()

    def _assign_uniform_positions(self):
        # Generate random points on a sphere:
        sin_ph, cos_ph = utils.general.sin_cos(np.random.uniform(0, 2 * np.pi, self.n), '0,2pi')
        cos_th = np.random.uniform(-1, 1, self.n)
        sin_th = np.sqrt(1 - cos_th * cos_th)
        self.x = sin_th * cos_ph
        self.y = sin_th * sin_ph
        self.z = cos_th

    def time_evolve(self, delta_t):
        delta_t = np.asarray(delta_t)
        assert delta_t.shape in ((), (1,), (self.n,))
        assert np.all(delta_t > 0)
        self.x, self.y, self.z = utils.diffusive_steps.safe_diffusive_step(
            self.x, self.y, self.z,
            normalized_time_step=delta_t/self.rot_diffusion_time)
        self.t += delta_t


class ElectronicState:
    """
    The ElectronicState class represents an electronic state of a molecule.

    :param name: The name of the electronic state.
    :type name: str
    :param lifetime: The lifetime of the electronic state.
    :type lifetime: float
    :param transition_states: The list of possible transition states for this electronic state.
    :type transition_states: List[str]
    :param probabilities: The list of transition probabilities for this electronic state.
    :type probabilities: List[float]

    Attributes:

    - name (str): The name of the electronic state.
    - lifetime (float): The lifetime of the electronic state.
    - transition_states (List[str]): The list of possible transition states for this electronic state.
    - probabilities (List[float]): The list of transition probabilities for this electronic state.
    - state_num (int): The number assigned to this electronic state.
    """
    def __init__(self, name: str, lifetime: float,
                 transition_states=None,
                 probabilities=None):
        self.name = name
        self.lifetime = lifetime
        self.transition_states = None
        self.probabilities = None
        self.state_num = None

        # validate input
        assert isinstance(self.name, str)
        assert isinstance(self.lifetime, (int, float))
        assert self.lifetime > 0
        self._validate_transition_states(transition_states)
        self._validate_probabilities(probabilities)

    def _validate_transition_states(self, transition_states):
        """Set default and clean up transition states"""
        if transition_states is None:
            transition_states = [self.name]
        if isinstance(transition_states, str):
            transition_states = [transition_states]
        for idx, transition_state in enumerate(transition_states):
            if not isinstance(transition_state, str):
                raise TypeError("transition states must be a string or list of strings")
        self.transition_states = transition_states

    def _validate_probabilities(self, probabilities):
        """Set default and clean up probabilities"""
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

    def assign_state_num(self, state_num):
        self.state_num = state_num


class PossibleStates:
    """The PossibleStates class represents a collection of possible electronic states for a molecule.

    :param electronic_state: The initial electronic state to add to the collection.
    :type electronic_state: ElectronicState

    Attributes:

    - dict (Dict[str, ElectronicState]): A dictionary of electronic states, with the state names as keys.
    - valid (bool): A flag indicating whether the collection of states is valid (i.e., all transition states exist in the collection).
    - orphan_states (List[Tuple[str, str]]): A list of tuples containing the state names of orphan states (i.e., states with transition states that do not exist in the collection).

    Methods:

    - add_state: Adds a new electronic state to the collection.
    - get_state_num_and_lifetime: Returns the state number and lifetime of the specified electronic states.
    """
    def __init__(self, electronic_state: ElectronicState):
        self.dict = {}
        self.valid = False
        self.orphan_states = []
        self.add_state(electronic_state)

    def add_state(self, electronic_state: ElectronicState):
        assert electronic_state.name not in self.dict
        electronic_state.assign_state_num(len(self.dict))
        self.dict[electronic_state.name] = electronic_state
        self._validate()

    def _validate(self):
        self.valid = True
        self.orphan_states = []
        for state_name, state in self.dict.items():
            for transition_state in state.transition_states:
                if transition_state not in self.dict.keys():
                    self.orphan_states.append((state_name, transition_state))
        if len(self.orphan_states) > 0:
            self.valid = False

    def get_state_num_and_lifetime(self, states):
        if not isinstance(states, list):
            if not (isinstance(states, int) or isinstance(states, str)):
                raise TypeError("states must be an integer, a string, or a list of either.")
            states = [states]
        state_nums = np.asarray([self[state].state_num for state in states], 'uint')
        lifetimes = np.asarray([self[state].lifetime for state in states], 'float')
        return state_nums, lifetimes

    def __getitem__(self, item):
        if not self.valid:
            raise LookupError(
                "\nPossibleStates is not usable yet.\n" +
                f"States {[state[0] for state in self.orphan_states]} "
                f"have unmatched final states {[state[1] for state in self.orphan_states]}, "
                f"which have not been set with .add().")
        if isinstance(item, str):
            return self.dict[item]
        elif isinstance(item, int):
            return list(self.dict.values())[item]
        else:
            raise TypeError("PossibleStates indices must be integers or strings.")


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



