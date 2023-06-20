from .base import PLModelWrapper

import torch
import pytorch_lightning as pl


class TestModelWrapper(PLModelWrapper):
    """
    This model wrapper is meant purely for testing purposes, to make sure that
    the distributed processing is working as expected
    """
    def __init__(self, model_dir, **kwargs):
        if 'm' not in kwargs:
            raise RuntimeError("Must specify 'm'. Use additional-kwargs")

        self._m = float(kwargs['m'])

        if 'b' not in kwargs:
            raise RuntimeError("Must specify 'b'. Use additional-kwargs")

        self._b = float(kwargs['b'])

        super().__init__(model_dir=model_dir, **kwargs)


    def load_model(self, model_path):
        self.model = TestModel(self._m, self._b)


    def compute_loss(self, batch):
        y = self.model(batch)

        return {'y': y}

    def aggregate_loss(self, step_outputs):
        all_outputs = torch.cat([s['y'] for s in step_outputs])
        all_outputs = self.all_gather(all_outputs).detach().cpu().numpy()

        self.results['e_rmse'] = all_outputs.sum()
        self.results['f_rmse'] = all_outputs.sum()


    def copy(self, model_path):
        pass


class TestModel(pl.LightningModule):
    """A dummy linear model: y = mx+b"""

    def __init__(self, m, b):
        super().__init__()

        self.m = torch.nn.Parameter(torch.Tensor([m]))
        self.b = torch.nn.Parameter(torch.Tensor([b]))

    def forward(self, x):
        return self.m*x + self.b
