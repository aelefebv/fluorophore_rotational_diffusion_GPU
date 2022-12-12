from dataclasses import dataclass

import numpy as np


@dataclass
class GFP:
    singlet_lifetime_ns: float = 3
    triplet_lifetime_ns: float = 6E06
    singlet_quantum_yield: float = 0.79
    triplet_quantum_yield: float = 0.01
    rotational_diffusion_time_ns: float = 23


@dataclass
class mScarlet:
    singlet_lifetime_ns: float = 3
    triplet_lifetime_ns: float = 5E05
    singlet_quantum_yield: float = 0.70
    triplet_quantum_yield: float = 0.01
    rotational_diffusion_time_ns: float = 1.6E05 * np.pi


@dataclass
class rsEGFP2:
    singlet_lifetime_ns: float = 5E05
    singlet_quantum_yield: float = 0.35
    rotational_diffusion_time_ns: float = 1E05 * np.pi  # figure S5 Ilaria's
