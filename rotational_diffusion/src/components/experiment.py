from rotational_diffusion.src.utils import state_info_creator
from rotational_diffusion.src.components import fluorophore


class Experiment:
    def __init__(self, molecule, num_molecules, triplet=False, photobleach=False):
        self.molecule = molecule
        self.num_molecules = num_molecules

        if triplet:
            state_info_function = state_info_creator.create_triplet_state_info
        else:
            state_info_function = state_info_creator.create_singlet_state_info
        self.state_info = state_info_function(self.molecule, photobleach)
        self.fluorophores = fluorophore.FluorophoreCollection(
            num_molecules=self.num_molecules,
            state_info=self.state_info,
            rot_diffusion_time=molecule.rotational_diffusion_time_ns
        )
