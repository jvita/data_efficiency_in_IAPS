from .base import PLModelWrapper

import ast
import torch

from dscribe.descriptors import ValleOganov

class ValleOganovModelWrapper(PLModelWrapper):
    def __init__(self, model_dir, **kwargs):
        if 'cutoff' not in kwargs:
            raise RuntimeError("Must specify cutoff distance for ValleOganov. Use --additional-kwargs argument.")

        self.cutoff = float(kwargs['cutoff'])

        if 'elements' not in kwargs:
            raise RuntimeError("Must specify elements for ValleOganov. Use --additional-kwargs argument.")

        self.elements = ast.literal_eval(kwargs['elements'])

        if 'pad_atoms' not in kwargs:
            self.pad_atoms = False
        else:
            self.pad_atoms = ast.literal_eval(kwargs['pad_atoms'])

        super().__init__(model_dir=model_dir, **kwargs)


    def load_model(self, model_path):
        self.model = ValleOganov(
            species=self.elements,
            k2={  # RDF settings
                "sigma": 10**(-0.5),  # gaussian smoothing width
                "n": 100,             # number of discretization points
                "r_cut": self.cutoff  # cutoff distance
            },
            k3={  # ADF settings
                "sigma": 10**(-0.5),
                "n": 100,
                "r_cut": self.cutoff
            },
        )


    def compute_structure_representations(self, batch):
        """
        Assumes that 'batch' is just a list of ASE Atoms objects with supercell
        energies stored under the `atoms.info['energy']` field
        """

        representations = []
        splits          = []
        energies        = []

        for original in batch:
            atoms = original.copy()

            natoms = len(atoms)

            if self.pad_atoms:
                atoms.center(vacuum=2*self.cutoff)
                atoms.pbc = True

            v = torch.from_numpy(self.model.create(atoms))

            representations.append(v)
            # representations.append(v.tile((natoms, 1)))

            splits.append(natoms)
            energies.append(atoms.info['energy']/natoms)

        # TODO: make sure .stack() didn't mess up the dimensions
        representations = torch.stack(representations, dim=0)
        energies        = torch.Tensor(energies)

        return {
            'representations': representations,
            'representations_energy': energies,
        }


    def copy(self, model_path):
        pass
