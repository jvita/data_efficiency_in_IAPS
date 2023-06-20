from .base import PLDataModuleWrapper

import os
import ast
import numpy as np

import schnetpack.transform as tform
from schnetpack.data import AtomsLoader, AtomsDataFormat
from schnetpack.data.datamodule import AtomsDataModule


class SchNetDataModule(PLDataModuleWrapper):
    def __init__(self, stage, **kwargs):
        """
        Arguments:

            stage (str):
                Path to a directory containing the following files:

                    * `full.db`: the formatted schnetpack.data.ASEAtomsData database
                    * `split.npz`: the file specifying the train/val/test split indices
        """
        if 'cutoff' not in kwargs:
            raise RuntimeError("Must specify cutoff distance for SchNetDataModule. Use --additional-kwargs argument.")

        self.cutoff = float(kwargs['cutoff'])

        if 'remove_offsets' not in kwargs:
            self.remove_offsets = True
        else:
            self.remove_offsets = ast.literal_eval(kwargs['remove_offsets'])

        if 'train_filename' in kwargs:
            self.train_filename = kwargs['train_filename']
        else:
            self.train_filename = None
        if 'test_filename' in kwargs:
            self.test_filename = kwargs['test_filename']
        else:
            self.test_filename = None
        if 'val_filename' in kwargs:
            self.val_filename = kwargs['val_filename']
        else:
            self.val_filename = None

        super().__init__(stage=stage, **kwargs)


    def setup(self, stage):
        """
        Populates the `self.train_dataset`, `self.test_dataset`, and
        `self.val_dataset` class attributes. Will be called automatically in
        __init__()
        """

        transforms = [
            tform.MatScipyNeighborList(cutoff=self.cutoff),
        ]

        if self.remove_offsets:
            transforms.insert(
                0,
                tform.RemoveOffsets('energy', remove_mean=True, remove_atomrefs=False)
            )

        datamodule = AtomsDataModule(
            datapath=os.path.join(stage, 'full.db'),
            split_file=os.path.join(stage, 'split.npz'),
            format=AtomsDataFormat.ASE,
            load_properties=['energy', 'forces'],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            transforms=transforms,
        )

        datamodule.setup()

        # TODO: allow optional loading of train/test/val files by name

        self.train_dataset  = datamodule.train_dataset
        self.test_dataset   = datamodule.test_dataset
        self.val_dataset    = datamodule.val_dataset


    def get_dataloader(self, dataset):
        return AtomsLoader(
                dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=False,
                # shuffle=True,
                # pin_memory=self._pin_memory,
            )





