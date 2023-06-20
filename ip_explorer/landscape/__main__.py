#!/usr/bin/env python
# coding: utf-8

# Imports

import logging

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)

import re
import copy
import random
import numpy as np

import torch
import torchmetrics

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI

import pytorch_lightning as pl

import loss_landscapes
import loss_landscapes.metrics
from loss_landscapes.model_interface.model_wrapper import SimpleModelWrapper, ModelWrapper
from loss_landscapes.model_interface.model_parameters import ModelParameters
from loss_landscapes.model_interface.torch.torch_wrappers import TorchModelWrapper
from loss_landscapes.model_interface.model_interface import wrap_model

from ip_explorer.datamodules import get_datamodule_wrapper
from ip_explorer.models import get_model_wrapper
from ip_explorer.landscape.loss import EnergyForceLoss

logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

parser = argparse.ArgumentParser(
    description="Generate loss landscapes"
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
parser.add_argument( '--model-path', type=str, help='Full path to model checkpoint file', dest='model_path', required=True,) 
parser.add_argument( '--layers-regex', type=str, help='RegEx string for selecting model layers by name', dest='layers_regex', default='.*',) 

parser.add_argument( '--landscape-type', type=str, help='Type of landscape to generate', dest='landscape_type', required=True, choices=['lines', 'plane'])
parser.add_argument( '--compute-initial-losses', help='Computes and logs the train/test/val losses of the inital model', action='store_true')
parser.add_argument( '--no-compute-initial-losses', action='store_false', dest='compute_initial_losses')
parser.set_defaults(compute_initial_losses=True)
parser.add_argument( '--compute-landscape', help='Computes the loss landscape', action='store_true')
parser.add_argument( '--no-compute-landscape', action='store_false', dest='compute_landscape')
parser.set_defaults(compute_landscape=True)

parser.add_argument( '--batch-size', type=int, help='Batch size for data loaders', dest='batch_size', default=128, required=False,)

parser.add_argument( '--loss-type', type=str, help='"energy", "force" or None', dest='loss_type', default=None, required=False,) 
parser.add_argument( '--aggregation-method', type=str, help='Method for aggregating over batches. Must be "rmse" or "max"', dest='aggregation_method', default="rmse", required=False,) 
parser.add_argument( '--distance', type=float, help='Fractional distance in parameterspace', dest='distance') 
parser.add_argument( '--steps', type=int, help='Number of grid steps in each direction in parameter space', dest='steps', required=True,) 
parser.add_argument( '--n-lines', type=int, help='Number of lines to evaluate if `landscape-type=="lines"`. Default is same as `steps`', dest='n_lines') 

parser.add_argument( '--additional-kwargs', type=str, help='A string of additional key-value argument pairs that will be passed to the model and datamodule wrappers. Format: "key1:value1 key2:value2"', dest='additional_kwargs', required=False, default='') 
parser.add_argument( '--additional-datamodule-kwargs', type=str, help='A string of additional key-value argument pairs that will be passed to the datamodule wrapper. Format: "key1:value1 key2:value2"', dest='additional_datamodule_kwargs', required=False, default='') 

args = parser.parse_args()

print('ALL ARGUMENTS:')
print(args)

# Seed RNGs
if args.seed is None:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    local_seed = np.random.randint(1000)

    global_seed = int(np.average(comm.allgather(local_seed)))

    args.seed = global_seed

    logging.info(f"Random seed must be consistent across all processes to ensure landscape correctness. Settting to {args.seed}")

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def main():

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Setup
    if args.landscape_type == 'lines':
        if args.n_lines is None:
            args.n_lines = args.steps

    if rank == 0:
        if not os.path.isdir(args.save_dir):
            os.makedirs(args.save_dir)

    comm.Barrier()  # wait for save directory to be created

    os.chdir(args.save_dir)

    additional_kwargs = {}
    for kv_pair in args.additional_kwargs.split():
        k, v = kv_pair.split(':')
        additional_kwargs[k] = v

    additional_datamodule_kwargs = {}
    for kv_pair in args.additional_datamodule_kwargs.split():
        k, v = kv_pair.split(':')
        additional_datamodule_kwargs[k] = v

    model = get_model_wrapper(args.model_type)(
        model_dir=args.model_path,
        copy_to_cwd=True,
        values_to_compute=['loss'],
        **additional_kwargs,
    )

    model.eval()

    datamodule = get_datamodule_wrapper(args.model_type)(
        args.database_path,
        batch_size=args.batch_size,
        # num_workers=int(np.floor(int(os.environ['LSB_MAX_NUM_PROCESSORS'])/int(args.gpus_per_node)/int(args.num_nodes))),
        num_workers=0,
        **additional_datamodule_kwargs,
    )

    if args.compute_initial_losses:
        if rank == 0:
            model.values_to_compute = ['energies_and_forces']

            # TODO: use devices=1 for train/test/val verification to avoid
            # duplicating data, as suggested on this page:
            # https://pytorch-lightning.readthedocs.io/en/stable/common/evaluation_intermediate.html
            trainer = pl.Trainer(
                num_nodes=1,
                devices=1,
                accelerator='cuda',
            )

            datasets_to_check = {
                'train': datamodule.train_dataloader(),
                # 'test': datamodule.test_dataloader(),
                # 'val': datamodule.val_dataloader(),
            }

            rmse_values = {}

            for name, dset in datasets_to_check.items():

                # Compute initial train/val/test losses
                print(f'Computing {name} errors with devices=1 to avoid batch padding errors', flush=True)

                trainer.test(model, dataloaders=dset)

                true_energies = model.results['true_energies']
                pred_energies = model.results['pred_energies']

                true_forces = model.results['true_forces']
                pred_forces = model.results['pred_forces']

                np.savetxt(
                    os.path.join(args.save_dir, args.prefix+f'true_{name}_energies_'+args.model_type+'.npy'),
                    true_energies,
                )

                np.savetxt(
                    os.path.join(args.save_dir, args.prefix+f'pred_{name}_energies_'+args.model_type+'.npy'),
                    pred_energies,
                )

                np.savetxt(
                    os.path.join(args.save_dir, args.prefix+f'true_{name}_forces_'+args.model_type+'.npy'),
                    true_forces,
                )

                np.savetxt(
                    os.path.join(args.save_dir, args.prefix+f'pred_{name}_forces_'+args.model_type+'.npy'),
                    pred_forces,
                )

                rmse_values[name] = {
                    'energy': np.sqrt(np.average((true_energies - pred_energies)**2)),
                    'forces': np.sqrt(np.average((true_forces - pred_forces)**2)),
                }

            print('E_RMSE (eV/atom), F_RMSE (eV/Ang)')
            print(f'\tTrain:\t{rmse_values["train"]["energy"]}, \t{rmse_values["train"]["forces"]}')
            # print(f'\tTest:\t{rmse_values["test"]["energy"]}, \t{rmse_values["test"]["forces"]}')
            # print(f'\tVal:\t{rmse_values["val"]["energy"]}, \t{rmse_values["val"]["forces"]}')

            model.values_to_compute = ['loss']

    comm.Barrier()  # make sure rank=0 is done with single-GPU calculation

    if not args.compute_landscape:
        return

    layers = []
    for k, p in model.named_parameters():
        if re.search(args.layers_regex, k):
            layers.append(k)

    model_final = SimpleModelWrapper(model, layers)  # needed for loss_landscapes

    print('MODEL PARAMETERS:')
    num_params = 0
    for k, p in model_final.named_parameters():
        print(k, p.shape)
        num_params += np.prod(p.shape)
    print('TOTAL NUM_PARAMS:', num_params)

    if args.distance is not None:
        distance = args.distance
    else:

        # Compute the distance to a random model
        init_model = model.random_model(model_path=args.model_path)

        # model_initial = SimpleModelWrapper(init_model)  # needed for loss_landscapes
        model_initial = SimpleModelWrapper(init_model, layers)  # needed for loss_landscapes

        start_point = copy.deepcopy(model_initial).get_module_parameters()
        end_point   = copy.deepcopy(model_final).get_module_parameters()

        with torch.no_grad():
            diff = ModelParameters([
                # torch.abs(end_point.parameters[i] - start_point.parameters[i])
                end_point.parameters[i] - start_point.parameters[i]
                for i in range(len(start_point.parameters))
            ])

            # Convert distance to units of multiples of model norm magnitude
            diff.truediv_(start_point.model_norm())
            # diff.filter_normalize_(end_point)

        distance = 2*diff.model_norm()
        logging.info(f"Distance not provided. Using 2x distance to random ({distance}).")


    # Switch to using a distributed model. Note that this means there will be
    # some noise in the generated landscapes due to batch padding.

    logging.info(f'[rank {rank}] Beginning LL generation')

    trainer = pl.Trainer(
        num_nodes=args.num_nodes,
        devices=args.gpus_per_node,
        accelerator='cuda',
        strategy='ddp',
        enable_progress_bar=False,
    )

    metric = EnergyForceLoss(
        evaluation_fxn=trainer.test,
        data_loader=datamodule.train_dataloader(),
        aggregation_method=args.aggregation_method,
    )

    if args.landscape_type == 'lines':
        loss_data_fin = loss_landscapes.random_line(
            model_final,
            metric,
            n_lines=args.n_lines,
            distance=distance,     # maximum (normalized) distance in parameter space
            steps=args.steps,           # number of steps
            normalization='filter',
            deepcopy_model=False,
            n_loss_terms=2,
        )

        loss_data_fin = np.transpose(loss_data_fin, axes=(2,0,1))

    elif args.landscape_type == 'plane':
        loss_data_fin = loss_landscapes.random_plane(
            model_final,
            metric,
            distance=distance,     # maximum (normalized) distance in parameter space
            steps=args.steps,           # number of steps
            normalization='filter',
            deepcopy_model=False,
            n_loss_terms=2,
        )

    if rank == 0:
        save_name = '{}={}_d={:.2f}_s={}_'.format(args.landscape_type, 'energy', distance, args.steps)
        if args.landscape_type == 'lines':
            save_name += f'{args.n_lines}_'
        full_path = os.path.join(args.save_dir, args.prefix+save_name+args.model_type)
        np.save(full_path, loss_data_fin[0])

        save_name = '{}={}_d={:.2f}_s={}_'.format(args.landscape_type, 'forces', distance, args.steps)
        if args.landscape_type == 'lines':
            save_name += f'{args.n_lines}_'
        full_path = os.path.join(args.save_dir, args.prefix+save_name+args.model_type)
        np.save(full_path, loss_data_fin[1])

        print("Saving results in:", args.save_dir)

        print('Done generating loss landscape!')


if __name__ == '__main__':
    os.environ['GPUS_PER_NODE'] = str(args.gpus_per_node)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    print(f'[rank={rank}] MASTER ADDR:', os.environ['MASTER_ADDR'])
    print(f'[rank={rank}] MASTER PORT:', os.environ['MASTER_PORT'])

    # Thanks to Adam T. Moody for helping me set this up!
    if 'OMPI_COMM_WORLD_RANK' in os.environ:
        os.environ["RANK"] = os.environ['OMPI_COMM_WORLD_RANK']
    if 'OMPI_COMM_WORLD_SIZE' in os.environ:
        os.environ["WORLD_SIZE"] = os.environ['OMPI_COMM_WORLD_SIZE']
    if 'OMPI_COMM_WORLD_LOCAL_RANK' in os.environ:
        os.environ["LOCAL_RANK"] = os.environ['OMPI_COMM_WORLD_LOCAL_RANK']

    print('OS ENVIRON:', os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'], os.environ['OMPI_COMM_WORLD_SIZE'], os.environ['OMPI_COMM_WORLD_RANK'], os.environ['LOCAL_RANK'])

    torch.distributed.init_process_group(
        backend="nccl", init_method="env://",
        world_size=int(os.environ['OMPI_COMM_WORLD_SIZE']),
        rank=int(os.environ['OMPI_COMM_WORLD_RANK'])
    )

    main()
