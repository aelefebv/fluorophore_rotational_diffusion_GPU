from rotational_diffusion.src import utils
import numpy as np

from rotational_diffusion.src.utils.general import sin_cos

class StateInfo:



class PossibleStates:
    """
    Holds a list of possible fluorophore states and initializes blank.
    """
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



