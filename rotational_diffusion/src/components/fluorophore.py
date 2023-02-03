from rotational_diffusion.src import utils, np


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
    def __init__(self, name: str, lifetime: float = np.inf,
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

    def __contains__(self, item):
        try:
            if self[item] is not None:
                return True
        except (KeyError, IndexError):
            return False


class FluorophoreCollection:
    """
    Generates a number of fluorophores with specified diffusion times and fluorophore states based on the
    FluorophoreStateInfo object. Contains methods to manipulate the fluorophores, such as
    simulating photo transitions, rotational diffusion, and fluorophore deletions.
    """
    def __init__(
            self,
            num_molecules,
            rot_diffusion_time,
            state_info: PossibleStates,
            orientation_initial='uniform',
            state_initial=0,
    ):
        assert isinstance(state_info, PossibleStates)
        assert state_initial in state_info

        self.state_info = state_info
        self.orientations = Orientations(num_molecules, rot_diffusion_time, orientation_initial)
        self.states = np.full(self.orientations.n, state_initial, dtype='uint8')
        self.transition_times = np.random.exponential(
            self.state_info[state_initial].lifetime, self.orientations.n
        )
        # The order of molecules isn't preserved, so we give them unique id's:
        self.id = np.arange(self.orientations.n, dtype='int')
        # We record molecular orientation and time for each spontaneous
        # transition. We use this information to simulate measurements,
        # since spontaneous transitions (e.g. excited->ground) are often
        # associated with emitting light:
        self.transition_events = {k: [] for k in ('x', 'y', 'z', 't', 'initial_state', 'final_state')}

    # todo should change this to take in some kind of excitation scheme?
    def phototransition(
        self,
        initial_state,  # Integer or string
        final_states,  # Integer/string or iterable of integers/strings
        state_probabilities=None,  # None, or array-like of floats
        intensity=1,  # Saturation units
        polarization_xyz=(0, 0, 1),  # Only the direction matters
    ):
        if len(self.id) == 0:
            return None  # No molecules, don't bother

        # Input sanitization
        assert initial_state in self.state_info
        initial_state = self.state_info[initial_state].state_num  # Ensure int
        final_states, lifetimes = self.state_info.get_state_num_and_lifetime(final_states)

        if final_states.shape == (1,):
            assert state_probabilities is None
        else:
            state_probabilities = np.asarray(state_probabilities, 'float')
            assert state_probabilities.shape == final_states.shape
            assert np.all(state_probabilities > 0)
            state_probabilities /= state_probabilities.sum()  # Sums to 1

        assert intensity > 0
        polarization_xyz = np.asarray(polarization_xyz, dtype='float')
        assert polarization_xyz.shape == (3,)
        polarization_xyz /= np.linalg.norm(polarization_xyz)  # Unit vector
        # A linearly polarized pulse of light, oriented in an arbitrary
        # direction, drives molecules to change their state. The
        # 'effective intensity' for each molecule varies like the square
        # of the cosine of the angle between the light's polarization
        # direction and the molecular orientation.
        i = (self.states == initial_state)  # Who's in the initial state?
        px, py, pz, = np.sqrt(intensity) * polarization_xyz
        o = self.orientations  # Temporary short nickname
        effective_intensity = (px*o.x[i] + py*o.y[i] + pz*o.z[i])**2  # Dot prod.
        selection_prob = 1 - 2**(-effective_intensity)  # Saturation units
        selected = np.random.uniform(0, 1, len(selection_prob)) <= selection_prob
        # Every photoselected molecule now changes to a new state. If
        # multiple 'final_states' are specified, the new state is
        # randomly selected according to 'state_probabilities'. New
        # 'transition_times' are randomly drawn for each new state from
        # an exponential distribution given by 'lifetimes'.
        t = o.t[i][selected]  # The current time
        tr_t = self.transition_times[i]  # A copy of relevant transition times
        if state_probabilities is None:
            self.states[i] = np.where(selected, final_states, self.states[i])
            tr_t[selected] = t + np.random.exponential(lifetimes, t.shape)
        else:
            which_state = np.random.choice(
                np.arange(len(final_states), dtype='int'),
                size=t.shape, p=state_probabilities)
            ss = self.states[i]  # A copy of the relevant states
            ss[selected] = final_states[which_state]
            self.states[i] = ss
            tr_t[selected] = t + np.random.exponential(lifetimes[which_state])
        self.transition_times[i] = tr_t

    # todo, this would be the "waits" step?
    def time_evolve(self, delta_t):
        if len(self.id) == 0:
            return None  # No molecules, don't bother
        assert delta_t > 0
        o = self.orientations  # Local nickname
        assert np.isclose(np.amin(o.t), np.amax(o.t)) # Orientations are synchronized
        target_time = o.t[0] + delta_t
        while np.any(o.t < target_time):
            # How much shall we step each molecule in time?
            dt = np.minimum(target_time, self.transition_times) - o.t
            idx = self._sort_by(dt)
            dt = dt if idx is None else dt[idx]  # Skip if dt is already sorted
            s = slice(np.searchsorted(dt, np.array(0), 'right'), None)  # Skip dt == 0
            # Update the orientations
            o.x[s], o.y[s], o.z[s] = utils.diffusive_steps.safe_diffusive_step(
                o.x[s], o.y[s], o.z[s], (dt/o.rot_diffusion_time)[s])
            o.t[s] += dt[s]
            # Calculate and record spontaneous transitions
            transitioning = (o.t >= self.transition_times)
            states = self.states[transitioning]  # Copy of states that change
            if states.size == 0:
                continue  # No states change; skip ahead.
            t = o.t[transitioning]
            self.transition_events['initial_state'].append(states)
            self.transition_events['t'            ].append(t)
            self.transition_events['x'            ].append(o.x[transitioning])
            self.transition_events['y'            ].append(o.y[transitioning])
            self.transition_events['z'            ].append(o.z[transitioning])
            idx = np.argsort(states)
            states = states[idx]  # A sorted copy of the states that change
            t = t[idx]
            transition_times = np.empty(len(states), dtype='float')
            for initial_state in range(len(self.state_info.dict)):
                s = slice(np.searchsorted(states, np.array(initial_state), 'left'),
                          np.searchsorted(states, np.array(initial_state), 'right'))
                if s.start == s.stop:
                    continue
                fs = self.state_info[initial_state].transition_states
                final_states, lifetimes = self.state_info.get_state_num_and_lifetime(fs)
                probabilities = self.state_info[initial_state].probabilities
                which_final = np.random.choice(
                    np.arange(len(final_states), dtype='int'),
                    size=int(s.stop-s.start), p=probabilities)
                states[s] = final_states[which_final]
                transition_times[s] = t[s] + np.random.exponential(lifetimes[which_final])
            # Undo our sorting of states and transition times, update originals
            idx_rev = np.empty_like(idx)
            idx_rev[idx] = np.arange(len(idx), dtype=idx.dtype)
            final_states = states[idx_rev]
            self.transition_events['final_state'].append(final_states)
            self.states[          transitioning] = final_states
            self.transition_times[transitioning] = transition_times[idx_rev]
        return None

    def get_xyz_for_state(self, state):
        assert state in self.state_info
        state = self.state_info[state].n  # Ensure int
        idx = (self.states == state)
        o = self.orientations  # Local nickname
        return o.x[idx], o.y[idx], o.z[idx]

    def get_xyzt_at_transitions(self, initial_state, final_state):
        assert initial_state in self.state_info
        assert final_state in self.state_info
        # We built 'self.transition_events' by appending 1D numpy arrays
        # to lists. Now's a good time to join each list of arrays into
        # a single array:
        if len(self.transition_events['t']) == 0: # No transitions yet
               return np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0)
        if len(self.transition_events['t']) > 1:
            for k, v in self.transition_events.items():
                self.transition_events[k] = [np.concatenate(v)]
        # Select only the records that correspond to a particular transition:
        e = self.transition_events # Local nickname
        tr = ((e['initial_state'][0] == self.state_info[initial_state].state_num) &
              (e[  'final_state'][0] == self.state_info[  final_state].state_num))
        x, y, z, t = e['x'][0][tr], e['y'][0][tr], e['z'][0][tr], e['t'][0][tr]
        return x, y, z, t

    def delete_fluorophores_in_state(self, state):
        if len(self.id) == 0:
            return None # No molecules, don't bother
        assert state in self.state_info
        state = self.state_info[state].state_num  # Convert to int
        idx = (self.states != state)
        self.states = self.states[idx]
        self.transition_times = self.transition_times[idx]
        o = self.orientations # Local nickname
        o.x, o.y, o.z, o.t = o.x[idx], o.y[idx], o.z[idx], o.t[idx]
        if o.rot_diffusion_time.shape == (o.n,):
            o.rot_diffusion_time = o.rot_diffusion_time[idx]
        o.n = len(o.t)
        self.id = self.id[idx]

    def _sort_by(self, x):
        x = np.asarray(x)
        assert x.shape == self.id.shape
        x_is_sorted = np.all(np.diff(x) >= 0)
        if x_is_sorted:
            return None
        idx = np.argsort(x)
        self.states = self.states[idx]
        self.transition_times = self.transition_times[idx]
        o = self.orientations  # Local nickname
        o.x, o.y, o.z, o.t = o.x[idx], o.y[idx], o.z[idx], o.t[idx]
        if o.rot_diffusion_time.size == o.n:
            o.rot_diffusion_time = o.rot_diffusion_time[idx]
        self.id = self.id[idx]
        return idx
