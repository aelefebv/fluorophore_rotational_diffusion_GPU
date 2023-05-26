from dataclasses import dataclass
from rotational_diffusion.src import np


@dataclass
class Fluorescein:
    singlet_lifetime_ns: float = 4
    singlet_quantum_yield: float = 1
    rotational_diffusion_time_ns: float = 3  #?

@dataclass
class GFP:
    singlet_lifetime_ns: float = 3
    triplet_lifetime_ns: float = 6E06
    singlet_quantum_yield: float = 0.79
    triplet_quantum_yield: float = 0.01
    rotational_diffusion_time_ns: float = 23


@dataclass
class AllSinglet:
    singlet_lifetime_ns: float = 5
    triplet_lifetime_ns: float = 1E06
    singlet_quantum_yield: float = 1
    triplet_quantum_yield: float = 0
    rotational_diffusion_time_ns: float = 100


@dataclass
class AllTriplet:
    singlet_lifetime_ns: float = 5
    triplet_lifetime_ns: float = 1E06
    singlet_quantum_yield: float = 0
    triplet_quantum_yield: float = 1
    rotational_diffusion_time_ns: float = 1E05


@dataclass
class mScarlet:
    rotational_diffusion_time_ns: float  # 1.6E05 * np.pi?
    singlet_lifetime_ns: float = 3
    triplet_lifetime_ns: float = 5E05
    singlet_quantum_yield: float = 0.70
    triplet_quantum_yield: float = 0.01


@dataclass
class mScarlet_high_triplet:
    rotational_diffusion_time_ns: float  # 1.6E05 * np.pi?
    singlet_lifetime_ns: float = 3
    triplet_lifetime_ns: float = 5E07
    singlet_quantum_yield: float = 0.70
    triplet_quantum_yield: float = 0.01


@dataclass
class mScarlet_low_triplet:
    rotational_diffusion_time_ns: float  # 1.6E05 * np.pi?
    singlet_lifetime_ns: float = 3
    triplet_lifetime_ns: float = 2.5E05
    singlet_quantum_yield: float = 0.70
    triplet_quantum_yield: float = 0.01


@dataclass
class rsEGFP2:
    singlet_lifetime_ns: float = 3
    singlet_quantum_yield: float = 0.35
    rotational_diffusion_time_ns: float = 23


@dataclass
class Venus:
    rotational_diffusion_time_ns: float  # 1.6E05 * np.pi?
    singlet_lifetime_ns: float = 2
    triplet_lifetime_ns: float = 1E06
    singlet_quantum_yield: float = 0.70
    triplet_quantum_yield: float = 0.01
