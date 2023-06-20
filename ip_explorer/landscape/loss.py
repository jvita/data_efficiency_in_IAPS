import numpy as np

import loss_landscapes
from loss_landscapes.model_interface.model_wrapper import ModelWrapper

from nequip.data import AtomicData


class EnergyForceLoss(loss_landscapes.metrics.Metric):
    """
    A wrapper for computing the energy/loss RMSE values for a given model and
    dataset. This class is specifically inteneded to interface with the
    loss_landscapes package.
    """

    data_loader    = None
    evaluation_fxn = None
    loss_type      = None

    def __init__(
        self,
        data_loader,
        evaluation_fxn=None,
        loss_type='both',
        aggregation_method='rmse',
        ):
        """
        Arguments:
            evaluation_fxn (callable):
                A function that takes as arguments `(model, data_loader)`,
                and returns a dictionary of values corresponding to chosen terms
                in the loss function. The only required dictionary keys are
                'energy' and 'force'.

            loss_type (str, default='both'):
                One of ['energy', 'force', 'both'].

            aggregation_method (str, default='rmse'):
                One of ['rmse', 'max']
        """
        super().__init__()
        self.data_loader      = data_loader
        self.evaluation_fxn   = evaluation_fxn
        self.loss_type        = loss_type
        self.aggregation_method = aggregation_method


    def __call__(self, model_wrapper: ModelWrapper) -> tuple:
        """
        Returns the computed loss terms. Returned units depend upon the
        specified `evaluation_fxn`, but are expected to be in eV/atom and
        eV/Ang.

        Returns:
            (energy_rmse,), (force_rmse,) or (energy_rmse, force_rmse)
        """
        self.evaluation_fxn(model_wrapper.modules[0], self.data_loader)

        if self.aggregation_method == 'rmse':
            loss_eng = model_wrapper.modules[0].results['e_rmse']
            loss_fcs = model_wrapper.modules[0].results['f_rmse']
        elif self.aggregation_method == 'max':
            loss_eng = model_wrapper.modules[0].results['e_max']
            loss_fcs = model_wrapper.modules[0].results['f_max']

        if self.loss_type == 'energy':
            return loss_eng
        elif self.loss_type == 'force':
            return loss_fcs
        else:
            return loss_eng, loss_fcs
