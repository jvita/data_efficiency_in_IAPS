from .base import PLDataModuleWrapper

import os
import logging
import numpy as np
from ase.io import read
from ast import literal_eval

from mace.tools import torch_geometric, get_atomic_number_table_from_zs
from mace.data import AtomicData, config_from_atoms_list


class MACEDataModule(PLDataModuleWrapper):
    def __init__(self, stage, **kwargs):
        """
        Arguments:

            stage (str):
                Path to a directory containing the following files:

                    * `train.xyz`: an ASE-readable list of atoms
                    * `test.xyz`: an ASE-readable list of atoms
                    * `val.xyz`: an ASE-readable list of atoms

            train_filename (str, default=None):
                If provided, used instead of `train.xyz`

            test_filename (str, default=None):
                If provided, used instead of `train.xyz`

            val_filename (str, default=None):
                If provided, used instead of `train.xyz`
        """

        if 'cutoff' not in kwargs:
            raise RuntimeError("Must specify cutoff distance for MACEDataModule.  Use --additional-datamodule-kwargs argument.")

        self.cutoff = float(kwargs['cutoff'])

        if 'train_filename' in kwargs:
            self.train_filename = kwargs['train_filename']
        else:
            self.train_filename = 'train.xyz'
        if 'test_filename' in kwargs:
            self.test_filename = kwargs['test_filename']
        else:
            self.test_filename = 'test.xyz'
        if 'val_filename' in kwargs:
            self.val_filename = kwargs['val_filename']
        else:
            self.val_filename = 'val.xyz'
        if 'z_table' in kwargs:
            self.z_table = get_atomic_number_table_from_zs(
                literal_eval(kwargs['z_table'])
            )
        else:
            self.z_table = None

        super().__init__(stage=stage, **kwargs)


    def setup(self, stage):
        """
        Populates the `self.train_dataset`, `self.test_dataset`, and
        `self.val_dataset` class attributes. Will be called automatically in
        __init__()
        """

        train_path = os.path.join(stage, self.train_filename)
        if os.path.isfile(train_path):
            train   = config_from_atoms_list(read(os.path.join(stage, self.train_filename), format='extxyz', index=':'))

            if self.z_table is None:
                z_table = get_atomic_number_table_from_zs(
                    z
                    for config in train
                    for z in config.atomic_numbers
                )
            else:
                z_table = self.z_table

            self.train_dataset  = [AtomicData.from_config(c, z_table=z_table, cutoff=self.cutoff) for c in train]
        else:
            logging.warning("File '{}' could not be loaded.".format(train_path))

        test_path = os.path.join(stage, self.test_filename)
        if os.path.isfile(test_path):
            test   = config_from_atoms_list(read(os.path.join(stage, self.test_filename), format='extxyz', index=':'))

            if self.z_table is None:
                z_table = get_atomic_number_table_from_zs(
                    z
                    for config in test
                    for z in config.atomic_numbers
                )
            else:
                z_table = self.z_table

            self.test_dataset   = [AtomicData.from_config(c, z_table=z_table, cutoff=self.cutoff) for c in test]
        else:
            logging.warning("File '{}' could not be loaded.".format(test_path))

        val_path = os.path.join(stage, self.val_filename)
        if os.path.isfile(val_path):
            val   = config_from_atoms_list(read(os.path.join(stage, self.val_filename), format='extxyz', index=':'))

            if self.z_table is None:
                z_table = get_atomic_number_table_from_zs(
                    z
                    for config in val
                    for z in config.atomic_numbers
                )
            else:
                z_table = self.z_table

            self.val_dataset    = [AtomicData.from_config(c, z_table=z_table, cutoff=self.cutoff) for c in val]
        else:
            logging.warning("File '{}' could not be loaded.".format(val_path))


    def get_dataloader(self, dataset):
        return torch_geometric.dataloader.DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=True,
        )
