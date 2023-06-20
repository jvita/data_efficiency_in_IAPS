from .base import PLModelWrapper

import ast
import torch

from dscribe.descriptors import SOAP

class SOAPModelWrapper(PLModelWrapper):
    """
    Parameters:

        cutoff (float):
            Radial cutoff distance

        elements (list):
            List of allowed element types (e.g., ["Al"])

        n_max (int):
            Number of radial basis functions

        l_max (int):
            Maximum degree of spherical harmonics

        pad_atoms (bool):
            If True, pads simulation cell with vacuum of size 2*`cutoff` along
            each dimension
    """

    def __init__(self, model_dir, **kwargs):
        if 'cutoff' not in kwargs:
            raise RuntimeError("Must specify cutoff distance for SOAP. Use --additional-kwargs argument.")

        self.cutoff = float(kwargs['cutoff'])

        if 'elements' not in kwargs:
            raise RuntimeError("Must specify elements for SOAP. Use --additional-kwargs argument.")

        self.elements = ast.literal_eval(kwargs['elements'])

        if 'n_max' not in kwargs:
            raise RuntimeError("Must specify n_max for SOAP. Use --additional-kwargs argument.")

        self.n_max = ast.literal_eval(kwargs['n_max'])

        if 'l_max' not in kwargs:
            raise RuntimeError("Must specify l_max for SOAP. Use --additional-kwargs argument.")

        self.l_max = ast.literal_eval(kwargs['l_max'])

        if 'pad_atoms' not in kwargs:
            self.pad_atoms = False
        else:
            self.pad_atoms = ast.literal_eval(kwargs['pad_atoms'])

        super().__init__(model_dir=model_dir, **kwargs)


    def load_model(self, model_path):
        self.model = soap = SOAP(
            species=self.elements,
            periodic=False,
            r_cut=self.cutoff,
            n_max=self.n_max,
            l_max=self.l_max,
        )


    def compute_atom_representations(self, batch):
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

            representations.append(torch.from_numpy(self.model.create(atoms)))

            splits.append(natoms)
            energies.append(torch.ones(natoms)*atoms.info['energy']/natoms)

        representations = torch.cat(representations, dim=0)
        energies        = torch.cat(energies)

        return {
            'representations': representations,
            'representations_splits': splits,
            'representations_energy': energies,
        }


    def copy(self, model_path):
        pass
