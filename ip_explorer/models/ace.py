from .base import PLModelWrapper

import os
import shutil
import torch
import numpy as np

from pyace.basis import BBasisConfiguration
from tensorpotential.potentials.ace import ACE
from tensorpotential.tensorpot import TensorPotential

class ACEModelWrapper(PLModelWrapper):
    """
    A wrapper for an ACE model.
    """
    def __init__(self, model_dir, **kwargs):
        # if 'representation_type' in kwargs:
        #     self.representation_type = kwargs['representation_type']
        # else:
        #     self.representation_type = 'node'


        super().__init__(model_dir=model_dir, **kwargs)


    def load_model(self, model_path):
        basis = BBasisConfiguration(os.path.join(model_path, 'interim_potential_0.yaml'))
        self.model = ACETorchWrapper(
            TensorPotential(ACE(basis))
        )


    def compute_loss(self, batch):

        natoms = torch.unique(torch.from_numpy(batch['cell_map']), return_counts=True)[1]

        true_eng = batch['energy']/natoms
        true_fcs = batch['forces']

        # loss, grad_loss, pred_eng, pred_fcs = self.model._tflow_model.native_fit(batch)
        loss, grad_loss, pred_eng, pred_fcs = self.model(batch)

        pred_eng = torch.from_numpy(pred_eng.numpy())/natoms
        pred_fcs = torch.from_numpy(pred_fcs.numpy())

        ediff = (pred_eng - true_eng).detach().cpu().numpy()
        fdiff = (pred_fcs - true_fcs).detach().cpu().numpy()

        return {
            'energy': np.mean(ediff**2),
            'force':  np.mean(fdiff**2),
            'batch_size': batch['energy'].shape[0],
            'natoms': sum(natoms).detach().cpu().numpy(),
        }


    def compute_atom_representations(self, batch):
        raise NotImplementedError

        out = self.model.forward(batch)

        with torch.no_grad():
            representations = []

            if self.representation_type in ['node', 'both']:
                representations.append(batch['scalar_representation'])

            if self.representation_type in ['edge', 'both']:

                if isinstance(self.model.representation, SchNet):
                    x = batch['scalar_representation']
                    z = x.new_zeros((batch['scalar_representation'].shape[0], self.model.radial_basis.n_rbf))

                    idx_i = batch[structure.idx_i]
                    idx_j = batch[structure.idx_j]

                    z.index_add_(0, idx_i, batch['distance_representation'])
                    z.index_add_(0, idx_j, batch['distance_representation'])
                else:  # PaiNN
                    z = batch['vector_representation']
                    z = torch.mean(z, dim=1)  # average over cartesian dimension

                representations.append(z)

        representations = torch.cat(representations, dim=1)

        return {
            'representations': representations,
            'representations_splits': batch['_n_atoms'].detach().cpu().numpy().tolist(),
            # 'representations_energy': batch_dict[AtomicDataDict.TOTAL_ENERGY_KEY],
            'representations_energy': batch['energy'],
        }



    def copy(self, model_path):
        shutil.copyfile(
            os.path.join(model_path, 'interim_potential_0.yaml'),
            os.path.join(os.getcwd(), 'interim_potential_0.yaml'),
        )


class ACETorchWrapper(torch.nn.Module):
    """
    This class wraps a TensorPotential implementation of ACE using hooks so that
    the underlying Tensorflow Variables can be exposed as PyTorch Parameters.
    """
    def __init__(self, tflow_model):
        super().__init__()

        self._tflow_model = tflow_model

        self.radial_coefs  = torch.nn.Parameter(torch.from_numpy(self._tflow_model.potential.fit_coefs[:self._tflow_model.potential.total_num_crad].numpy()))
        self.basis_coefs   = torch.nn.Parameter(torch.from_numpy(self._tflow_model.potential.fit_coefs[self._tflow_model.potential.total_num_crad:].numpy()))

        self.register_forward_pre_hook(self._tensorflow_copy_hook)


    @staticmethod
    def _tensorflow_copy_hook(module, inputs):
        module._tflow_model.potential.set_coefs(
            np.concatenate([
                module.radial_coefs.detach().cpu().numpy(),
                module.basis_coefs.detach().cpu().numpy(),
            ])
        )


    def forward(self, batch):
        return self._tflow_model.native_fit(batch)
