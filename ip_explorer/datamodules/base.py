from typing import Optional
import pytorch_lightning as pl


class PLDataModuleWrapper(pl.LightningDataModule):
    """
    A basic wrapper satisfying the expected structure of a
    `pl.LightningDataModule` object. When implementing a new DataModule, the
    only functions that should be implemented are `setup()` and
    `get_dataloader()`.
    """

    def __init__(self, stage, batch_size, num_workers, collate_fn=None, **kwargs):
        super().__init__()
        self.batch_size     = batch_size
        self.num_workers    = num_workers
        self.collate_fn     = collate_fn

        self.train_dataset = self.test_dataset = self.val_dataset = None
        self.setup(stage)

        # if self.train_dataset is None:
        #     raise RuntimeError("Failed to load training dataset. Make sure that `self.train_dataset` is assigned in `self.setup()`")
        # if self.test_dataset is None:
        #     raise RuntimeError("Failed to load testing dataset. Make sure that `self.test_dataset` is assigned in `self.setup()`")
        # if self.val_dataset is None:
        #     raise RuntimeError("Failed to load validation dataset. Make sure that `self.val_dataset` is assigned in `self.setup()`")


    def setup(self, stage: Optional[str] = None):
        """
        Populates the `self.train_dataset`, `self.test_dataset`, and
        `self.val_dataset` class attributes. Will be called automatically in
        __init__()
        """
        raise NotImplementedError

    def get_dataloader(self, dataset):
        raise NotImplementedError

    def train_dataloader(self):
        return self.get_dataloader(self.train_dataset)

    def test_dataloader(self):
        return self.get_dataloader(self.test_dataset)

    def val_dataloader(self):
        return self.get_dataloader(self.val_dataset)


