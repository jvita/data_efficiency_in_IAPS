from .base import PLDataModuleWrapper

import os
from torch.utils.data import DataLoader

from ase.io import read

class ASEDataModule(PLDataModuleWrapper):
    def __init__(self, stage, fmt='extxyz', **kwargs):
        """
        Arguments:

            stage (str):
                Full path to a file that is readable with `ase.io.read`. Only
                supports loading a single file, which will be used for training,
                testing, AND validation.

            fmt (str, default='extxyz'):
                File format specification to be passed to `ase.io.read`
        """
        self.fmt = fmt

        super().__init__(stage=stage, **kwargs)


    def setup(self, stage):
        self.train_dataset  = read(os.path.join(stage, 'train.xyz'), format=self.fmt, index=':')
        self.test_dataset   = read(os.path.join(stage, 'test.xyz'), format=self.fmt, index=':')
        self.val_dataset    = read(os.path.join(stage, 'val.xyz'), format=self.fmt, index=':')


    def get_dataloader(self, dataset):

        def collate_fn(samples):
            return samples

        return DataLoader(
                dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=False,
                collate_fn=collate_fn,
            )


