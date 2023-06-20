from .base import PLModelWrapper

import os
import torch
import shutil

from nequip.data import AtomicData, AtomicDataDict
from nequip.train import Trainer, Metrics
from nequip.model import model_from_config
from nequip.utils import Config, instantiate


class NequIPModelWrapper(PLModelWrapper):
    """
    A wrapper to a nequip model. Assumes that `traindir` contains a
    configuration file with the name 'config.yaml', and a model checkpoint with
    the name 'best_model.pth'.

    Note that the 'config.yaml' file should include the following keys:

        ```
        - - forces
          - rmse
        - - total_energy
          - rmse
          - PerAtom: true
        ```

    """
    def __init__(self, model_dir, **kwargs):
        if 'representation_type' in kwargs:
            self.representation_type = kwargs['representation_type']
        else:
            self.representation_type = 'node'

        if 'model_file_name' in kwargs:
            self.model_file_name = kwargs['model_file_name']
        else:
            self.model_file_name = 'best_model.pth'

        super().__init__(model_dir=model_dir, **kwargs)


    def load_model(self, traindir):

        self.model, config = Trainer.load_model_from_training_session(
            traindir=traindir,
            model_name=self.model_file_name
        )

        if ('structure_representations' in self.values_to_compute) or ('structure_representations' in self.values_to_compute):
            self._register_representations_hook()

        metrics_components = config.get("metrics_components", None)

        if metrics_components is None:
            # Default metrics
            metrics_components = [
                ['forces', 'mae'],
                ['forces', 'rmse'],
                ['total_energy', 'mae',  {'PerAtom': False}],
                ['total_energy', 'mae',  {'PerAtom': True}],
                ['total_energy', 'rmse', {'PerAtom': False}],
                ['total_energy', 'rmse', {'PerAtom': True}]
            ]

        metrics, _ = instantiate(
            builder=Metrics,
            prefix="metrics",
            positional_args=dict(components=metrics_components),
            all_args=config,
        )

        self.metrics = metrics

    def random_model(self, model_path):
        config = Config.from_file(model_path + "/config.yaml")

        return model_from_config(
            config=config,
            initialize=False,
        )

    def compute_energies_and_forces(self, batch):
        batch_dict = AtomicData.to_AtomicDataDict(batch)
        out = self.model.forward(batch_dict)

        natoms = torch.unique(out['batch'], return_counts=True)[1]

        return {
            'true_energies': batch_dict['total_energy']/natoms,
            'pred_energies': out['total_energy']/natoms,
            'true_forces': batch_dict['forces'],
            'pred_forces': out['forces'],
        }


    def compute_loss(self, batch):
        self.metrics.reset()

        batch_dict = AtomicData.to_AtomicDataDict(batch)

        out = self.model.forward(batch_dict)

        with torch.no_grad():
            self.metrics(out, batch)

        results, _ = self.metrics.flatten_metrics(self.metrics.current_result())

        if 'e/N_rmse' not in results:
            raise RuntimeError("e/N_rmse not found in metrics dictionary. Make sure the config file stored in self.model_path has the correct settings")

        if 'f_rmse' not in results:
            raise RuntimeError("f_rmse not found in metrics dictionary. Make sure the config file stored in self.model_path has the correct settings")

        return {
            'energy': results['e/N_rmse']**2,  # mse isn't implemented yet in nequip
            'force':  results['f_rmse']**2,
            # 'energy': results['e_mae'],  # mse isn't implemented yet in nequip
            # 'force':  results['f_mae'],
            'batch_size': int(max(batch_dict[AtomicDataDict.BATCH_KEY])+1),
            'natoms': batch_dict[AtomicDataDict.FORCE_KEY].shape[0],
        }


    def _register_representations_hook(self):
        """Add hook for extracting the output of the final convolution layer"""
        def hook(model, inputs):
            inputs[0]['node_representations'] = inputs[0][AtomicDataDict.NODE_FEATURES_KEY].clone()
            inputs[0]['edge_representations'] = inputs[0][AtomicDataDict.EDGE_EMBEDDING_KEY].clone()

        for name, module in self.model.named_modules():
            # if name.split('.')[-1] == 'conv_to_output_hidden':
            if name.split('.')[-1] == 'output_hidden_to_scalar':
                module.register_forward_pre_hook(hook)


    def compute_atom_representations(self, batch):
        batch_dict = AtomicData.to_AtomicDataDict(batch)

        out = self.model.forward(batch_dict)

        with torch.no_grad():
            z = out['node_representations']
            per_atom_representations = []

            if self.representation_type in ['node', 'both']:
                per_atom_representations.append(z)

            if self.representation_type in ['edge', 'both']:

                idx_i = out[AtomicDataDict.EDGE_INDEX_KEY][0, :]
                idx_j = out[AtomicDataDict.EDGE_INDEX_KEY][1, :]

                z_edge = z.new_zeros((z.shape[0], out['edge_representations'].shape[1]))
                z_edge.index_add_(0, idx_i, out['edge_representations'])
                z_edge.index_add_(0, idx_j, out['edge_representations'])

                per_atom_representations.append(z_edge)

        per_atom_representations = torch.cat(per_atom_representations, dim=1)
        splits = torch.unique(out['batch'], return_counts=True)[1]

        per_atom_energies = batch_dict[AtomicDataDict.TOTAL_ENERGY_KEY]/splits[:, None]
        per_atom_energies = torch.cat([
            per_atom_energies.new_ones(n)*e for n,e in zip(splits, per_atom_energies)
        ])

        return {
            'representations': per_atom_representations,
            'representations_splits': splits.detach().cpu().numpy().tolist(),
            'representations_energy': per_atom_energies,
        }


    def copy(self, traindir):
        shutil.copyfile(
            os.path.join(traindir, 'best_model.pth'),
            os.path.join(os.getcwd(), 'best_model.pth'),
        )

        shutil.copyfile(
            os.path.join(traindir, 'config.yaml'),
            os.path.join(os.getcwd(), 'config.yaml'),
        )
