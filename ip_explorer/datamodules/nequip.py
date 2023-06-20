from .base import PLDataModuleWrapper

import os

from nequip.train import Trainer
from nequip.utils import Config
from nequip.data import dataset_from_config
from nequip.data.dataloader import DataLoader


class NequIPDataModule(PLDataModuleWrapper):

    def __init__(self, stage, **kwargs):
        """
        Arguments:

            stage (str):
                The working directory used for training a NequIP model. Should
                contain the following files:

                    * `config.yaml`: the full configuration file, specifying data 
                        splits. Note that the configuration file should have the
                        'dataset', 'test_dataset', and 'validation_dataset'
                        keys.
        """

        if 'root' in kwargs:
            self.dataset_root = kwargs['root']
        else:
            self.dataset_root = None

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
        Arguments:

            stage (str):
                The working directory used for training a NequIP model. Should
                contain the following files:

                    * `config.yaml`: the full configuration file, specifying data 
                        splits. Note that the configuration file should have the
                        'dataset', 'test_dataset', and 'validation_dataset'
                        keys.
        """
        _, model_config = Trainer.load_model_from_training_session(traindir=stage)

        dataset_config = Config.from_file(
            os.path.join(stage, 'config.yaml'),
            defaults={"r_max": model_config["r_max"]}
        )

        # Check if different filename was given
        if self.train_filename is not None:
            dataset_config['dataset_file_name'] = self.train_filename
        if self.test_filename is not None:
            dataset_config['test_dataset_file_name'] = self.test_filename
        if self.val_filename is not None:
            dataset_config['validation_dataset_file_name'] = self.val_filename

        if self.dataset_root is not None:
            print('Loading processed dataset', flush=True)
            dataset_config['dataset_extra_fixed_fields']['root'] = self.dataset_root

        if 'dataset' in dataset_config:
            self.train_dataset  = dataset_from_config(dataset_config, prefix="dataset")

        if 'test_dataset' in dataset_config:
            self.test_dataset   = dataset_from_config(dataset_config, prefix="test_dataset")

        if 'validation_dataset' in dataset_config:
            self.val_dataset    = dataset_from_config(dataset_config, prefix="validation_dataset")

    def get_dataloader(self, dataset):
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
