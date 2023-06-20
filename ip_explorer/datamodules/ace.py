from .base import PLDataModuleWrapper

import os
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from pyace.preparedata import get_fitting_dataset

from tensorpotential.utils.utilities import batching_data


class ACEDataModule(PLDataModuleWrapper):
    def __init__(self, stage, **kwargs):
        """
        Arguments:

            stage (str):
                Path to a directory containing the following files:

                    * `train.gzip`: a zipped pandas DataFrame with the format described by the pyace documentation
                    * `test.gzip`: a zipped pandas DataFrame with the format described by the pyace documentation
                    * `val.gzip`: a zipped pandas DataFrame with the format described by the pyace documentation
        """

        if 'cutoff' not in kwargs:
            raise RuntimeError("Must specify cutoff distance for SchNetDataModule. Use --additional-kwargs argument.")

        self.cutoff = float(kwargs['cutoff'])

        super().__init__(stage=stage, **kwargs)


    def setup(self, stage):
        """
        Populates the `self.train_dataset`, `self.test_dataset`, and
        `self.val_dataset` class attributes. Will be called automatically in
        __init__()
        """

        self.train_dataset  = ACEDataSet(stage, 'train.gzip', cutoff=self.cutoff)
        self.test_dataset   = ACEDataSet(stage, 'test.gzip', cutoff=self.cutoff)
        self.val_dataset    = ACEDataSet(stage, 'val.gzip', cutoff=self.cutoff)


    def get_dataloader(self, dataset):

        def collate_fn(samples):
            return batching_data(pd.DataFrame(samples), batch_size=self.batch_size)[0]

        return DataLoader(
                dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=False,
                collate_fn=collate_fn,
            )


class ACEDataSet:
    def __init__(self, datapath, filename, cutoff):
        self._df = get_fitting_dataset(
            evaluator_name='tensorpot',
            # data_config=pd.read_pickle(path, compression='gzip'),
            data_config={
                'datapath': datapath,
                'filename': filename,
            },
            cutoff=cutoff,
        )

    def __len__(self):
        return len(self._df)

    def __getitem__(self, idx):
        return self._df.iloc[idx]
