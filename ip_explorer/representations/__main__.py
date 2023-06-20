#!/usr/bin/env python
# coding: utf-8

# Imports

import random
import numpy as np

import torch
import torchmetrics

import os
import argparse
import numpy as np

from ase import Atoms
from ase.io import write

import pytorch_lightning as pl

import loss_landscapes
import loss_landscapes.metrics
from loss_landscapes.model_interface.model_wrapper import SimpleModelWrapper

from ip_explorer.datamodules import get_datamodule_wrapper
from ip_explorer.models import get_model_wrapper

import logging
# logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

parser = argparse.ArgumentParser(
    description="Generate PES"
)

# Add CLI arguments
parser.add_argument( '--seed', type=int, help='The random seed to use', dest='seed', default=None, required=False,)
parser.add_argument( '--num-nodes', type=int, help='The number of nodes available for training', dest='num_nodes', required=True,) 
parser.add_argument( '--gpus-per-node', type=int, help='The number of GPUs per node to use', dest='gpus_per_node', default=1, required=False,) 

parser.add_argument( '--prefix', type=str, help='Prefix to add to the beginning of logged files', dest='prefix', default='', required=False)
parser.add_argument( '--save-dir', type=str, help='Directory in which to save the results. Created if does not exist', dest='save_dir', required=True)
parser.add_argument( '--overwrite', help='Allows save_directory to be overwritten', action='store_true')
parser.add_argument( '--no-ovewrite', action='store_false', dest='overwrite')
parser.set_defaults(overwrite=False)

parser.add_argument( '--model-type', type=str, help='Type of model being used.  Must be one of the supported types from ip_explorer', dest='model_type', required=True)
parser.add_argument( '--database-path', type=str, help='Path to formatted schnetpack.data.ASEAtomsData database', dest='database_path', required=True)
parser.add_argument( '--model-path', type=str, help='Full path to model checkpoint file', dest='model_path', default='.') 

parser.add_argument( '--batch-size', type=int, help='Batch size for data loaders', dest='batch_size', default=128, required=False,)
parser.add_argument( '--slice', type=int, help='Step size to use when reducing data size via slicing', default=1, dest='slice', required=False,)

parser.add_argument( '--additional-kwargs', type=str, help='A string of additional key-value argument pairs that will be passed to the model and datamodule wrappers. Format: "key1:value1 key2:value2"', dest='additional_kwargs', required=False, default='') 

args = parser.parse_args()

print('ALL ARGUMENTS:')
print(args)

# Seed RNGs
if args.seed is None:
    args.seed = np.random.randint()

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def main():

    if 'SHEAP_PATH' not in os.environ:
        print("'SHEAP_PATH' environment variable not found. Will only perform pre-processing steps")

    # Setup
    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)

    os.chdir(args.save_dir)

    additional_kwargs = {}
    for kv_pair in args.additional_kwargs.split():
        k, v = kv_pair.split(':')
        additional_kwargs[k] = v

    model = get_model_wrapper(args.model_type)(
        model_dir=args.model_path,
        values_to_compute=('atom_representations',),
        copy_to_cwd=True,
        **additional_kwargs,
    )

    model.eval()

    datamodule = get_datamodule_wrapper(args.model_type)(
        args.database_path,
        batch_size=args.batch_size,
        num_workers=int(np.floor(int(os.environ['LSB_MAX_NUM_PROCESSORS'])/int(os.environ['GPUS_PER_NODE']))),
        **additional_kwargs,
    )

    trainer = pl.Trainer(
        num_nodes=args.num_nodes,
        devices=args.gpus_per_node,
        accelerator='cuda',
        strategy='ddp',
        # enable_progress_bar=False,
    )

    trainer.test(model, dataloaders=datamodule.train_dataloader())

    representations = model.results['representations'].detach().cpu().numpy()
    representations_energies = model.results['representations_energies'].detach().cpu().numpy()
    representations_splits = model.results['representations_splits'].detach().cpu().numpy()

    print('REPRESENTATIONS SHAPE:', representations.shape)

    if representations.shape[0] != representations_energies.shape[0]:
        raise RuntimeError(f"# of representations ({representations.shape[0]}) != # of energies ({representations_energies.shape[0]}).")

    split_cumsum = np.cumsum(representations_splits)[:-1].astype(int)
    representations = np.array_split(representations, split_cumsum)
    representations_energies = np.array_split(representations_energies, split_cumsum)

    images = []
    for i, (v, e) in enumerate(zip(representations, representations_energies)):
        natoms = v.shape[0]
        atoms  =  Atoms(
            f'H{natoms}',
            positions=np.zeros((natoms, 3)),
            cell=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        )

        atoms.arrays['representations'] = v
        atoms.arrays['representations_energies'] = e

        # # SHEAP searches for "energy"
        # atoms.arrays['energy'] = e

        # # SHEAP searches for "SOAP*" keys
        # atoms.info['SOAP-n8-l4-c2.4-g0.3'] = v

        # # Filler information for SHEAP testing
        # atoms.info['name'] = str(i)
        # atoms.info['pressure']    = 0.0
        # atoms.info['spacegroup']  = 'unknown'
        # atoms.info['times_found'] = 1

        images.append(atoms)

    write(os.path.join(args.save_dir, args.prefix+'representations.xyz'), images, format='extxyz')

    print("Saving results in:", args.save_dir)

    if 'SHEAP_PATH' not in os.environ:
        return
    else:
        raise RuntimeError("SHEAP execution not supported yet. Perform SHEAP processing externally.")


if __name__ == '__main__':
    os.environ['GPUS_PER_NODE'] = str(args.gpus_per_node)
    main()
