from .base import PLDataModuleWrapper

import os
import torch
from torch.utils.data import DataLoader

class TestDataModule(PLDataModuleWrapper):
    """A dummy datamodule used for debugging purposes"""

    def __init__(self, stage, num_samples=100, sample_size=10, **kwargs):
        self.num_samples = num_samples
        self.sample_size = sample_size

        super().__init__(stage=stage, **kwargs)


    def setup(self, stage):
        self.train_dataset  = [torch.ones(self.sample_size) for _ in range(self.num_samples)]
        self.test_dataset   = [torch.ones(self.sample_size) for _ in range(self.num_samples)]
        self.val_dataset    = [torch.ones(self.sample_size) for _ in range(self.num_samples)]


    def get_dataloader(self, dataset):

        def collate_fn(samples):
            return torch.cat(samples)

        return DataLoader(
                dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=False,
                collate_fn=collate_fn,
            )


