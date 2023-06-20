import torch
import numpy as np
import pytorch_lightning as pl


class PLModelWrapper(pl.LightningModule):
    """
    A model wrapper for facilitating distributed execution via Pytorch
    Lightning. When implementing a new model that sub-classes from PLModelWrapper,
    the three functions that should be implemented are `load_model()`,
    `compute_loss()`, and `copy()`. Note that the default `aggregate_loss()`
    function will correctly aggregate MSE or MAE results; if `compute_loss()`
    does not return these results, then you should overload `aggregate_loss()`
    with a proper aggregation function.

    This wrapper utilizes the `pl.Trainer.test()` function as a workaround for
    distributed inference. In order to enable the distributed
    evaluation/aggregation of arbitrary results, users can define `compute_*()`
    and `aggregate_*()` functions, then utilize the `values_to_compute`
    constructor argument to specify which values should be computed during
    `test_step()` and aggregated during `test_epoch_end()`. At the very least,
    `compute_loss()` and `aggregate_loss()` should be implemented. See
    documentation in `compute_loss()` and `aggregate_loss()` for more details
    regarding how `compute_*()` and `aggregate_*()` functions should be written.

    NOTE: When doing distributed evaluation, Pytorch Lightning may pad the
    batches to ensure that the batch size is consistent across devices. This can
    lead to slightly incorrect statistics. Therefore, `devices=1` should be
    passed to the pl.Trainer class when exact statistics are required using this
    model.
    """

    def __init__(
        self,
        model_dir,
        values_to_compute=None,
        reset_results_on_epoch_start=True,
        copy_to_cwd=False,
        **kwargs
    ):
        """
        Arguments:

            model_dir (str):
                The path to a folder containing the information necessary to
                load the model. Will be passed to `PLModelWrapper.load_model` and
                `PLModelWrapper.copy()`

            values_to_compute (tuple, default=('loss',)):
                A tuple of strings specifying which values should be computed
                during test_step() and aggregated during test_epoch_end(). For
                each `value` in `values_to_compute`, the class functions
                `compute_{value}` and `aggregate_{value}` must be defined.

            reset_results_on_epoch_start (bool, default=True):
                If True, resets the contents of `self.results` at the beginning
                of every test epoch.

            copy_to_cwd (bool):
                If True, calls the `PLModelWrapper.copy()` function during
                instantiation.

        """
        super().__init__()

        if values_to_compute is None:
            self.values_to_compute = ('loss',)
        else:
            self.values_to_compute = tuple(values_to_compute)

        # Other administrative tasks
        self.results = {}
        self.reset_results_on_epoch_start = reset_results_on_epoch_start

        if copy_to_cwd:
            self.copy(model_dir)

        self.model_dir = model_dir

        # Load model at the end
        self.model = None
        self.load_model(model_dir)

        if self.model is None:
            raise RuntimeError("Failed to load model. Make sure to implement `load_model()` and assign `self.model`")


    def load_model(self, model_dir):
        """
        Populates the `self.model` variable with a callable PyTorch module. This
        is also where any additional model-specific class variables should be
        populated.
        """
        raise NotImplementedError

    def random_model(self):
        """
        Returns a model with parameters that have been randomly initialized
        in a manner similar to what would have been used during training.
        """
        raise NotImplementedError


    def compute_loss(self, batch):
        """
        A function for computing the energy and force MSE of the model. This
        function will be called by default during `test_step()` unless
        `self.values_to_compute` was specified to not contain "loss".

        When implementing new `compute_*()` functions, make sure that they
        return a dictionary that contains any keys that are necessary for the
        corresponding `aggregate_*()` function. For example, the
        `aggregate_loss()` function by default expects the following keys:
        "energy", "force", "batch_size", and "natoms".

        Arguments:

            batch:
                An object that can be passed directly to the `model.forward()`
                function.

        Returns:

            results (dict):
                A dictionary with four required keys:

                    * `'energy'`: the batch energy MSE (units of energy per atom)
                    * `'force'`: the batch force MSE (units of energy per distance)
                    * `'batch_size'`: the number of structures in the batch.
                        Used for computing weighted averages of energy errors.
                    * `'natoms'`: the number of atoms in the batch.
                        Used for computing weighted averages of force errors.
        """
        raise NotImplementedError


    def aggregate_loss(self, step_outputs):
        """
        A function that takes a list of outputs from `test_step()`, aggregates
        the results, and stores them under `self.results`.

        When implementing other `aggregate_*()` functions, make sure to store
        the results under `self.results` rather than returning the values. Also
        note that you should likely call `self.all_gather` in order to collect
        the results from all sub-processes.

        Arguments:

            step_outputs (list):
                A list of outputs returned by `test_step()`.

        Returns:

            None. Results must be stored under `self.results`.
        """

        # Note: this will return the maximum batch-averaged value.
        self.results['e_max'] = np.max([s['energy'] for s in step_outputs])
        self.results['f_max'] = np.max([s['force'] for s in step_outputs])

        # compute_loss MUST return the MSE or MAE so weighted aggregation is correct
        e_rmse = torch.Tensor([s['energy']*s['batch_size'] for s in step_outputs])
        f_rmse = torch.Tensor([s['force']*s['natoms'] for s in step_outputs])

        # Cast to int in case a length (1,) array was returned instead
        batch_sizes = sum([int(s['batch_size']) for s in step_outputs])
        natoms      = sum([int(s['natoms']) for s in step_outputs])

        e_rmse = self.all_gather(e_rmse)
        f_rmse = self.all_gather(f_rmse)

        batch_sizes = self.all_gather(batch_sizes)
        natoms      = self.all_gather(natoms)

        n_e_tot = batch_sizes.sum()
        n_f_tot = natoms.sum()

        e_rmse = np.sqrt((e_rmse/n_e_tot).sum().detach().cpu().numpy())
        f_rmse = np.sqrt((f_rmse/n_f_tot).sum().detach().cpu().numpy())

        self.results['e_rmse'] = e_rmse
        self.results['f_rmse'] = f_rmse


    def compute_energies(self, batch):
        raise NotImplementedError


    def compute_structure_representations(self, batch):

        retvals = self.compute_atom_representations(batch)

        representations = [
            torch.mean(_, dim=0)
            for _ in torch.split(
                retvals['representations'], retvals['representations_splits']
            )
        ]

        energies = [
            torch.sum(_, dim=0)
            for _ in torch.split(
                retvals['representations_energy'],
                retvals['representations_splits']
            )
        ]

        return {
            'representations': representations,
            'representations_splits': splits,
            'representations_energy': energies,
        }

    def compute_atom_representations(self, batch):
        raise NotImplementedError


    def aggregate_energies_and_forces(self, step_outputs):
        # On each worker, compute the per-structure average representations
        true_energies = []
        pred_energies = []
        true_forces = []
        pred_forces = []
        for s in step_outputs:
            true_energies.append(s['true_energies'])
            pred_energies.append(s['pred_energies'])
            true_forces.append(s['true_forces'])
            pred_forces.append(s['pred_forces'])

        true_energies = torch.cat(true_energies)
        pred_energies = torch.cat(pred_energies)
        true_forces = torch.cat(true_forces)
        pred_forces = torch.cat(pred_forces)

        # Now gather everything
        true_energies = self.all_gather(true_energies)
        pred_energies = self.all_gather(pred_energies)
        true_forces = self.all_gather(true_forces)
        pred_forces = self.all_gather(pred_forces)

        # Reshape to remove the num_processes dimension.
        # NOTE: order is likely not going to match dataloader order
        true_energies = torch.flatten(true_energies, 0, -1)
        pred_energies = torch.flatten(pred_energies, 0, -1)
        true_forces = torch.flatten(true_forces, 0, -1)
        pred_forces = torch.flatten(pred_forces, 0, -1)

        self.results['true_energies'] = true_energies.detach().cpu().numpy()
        self.results['pred_energies'] = pred_energies.detach().cpu().numpy()
        self.results['true_forces'] = true_forces.detach().cpu().numpy()
        self.results['pred_forces'] = pred_forces.detach().cpu().numpy()


    def aggregate_structure_representations(self, step_outputs):
        """
        Expects that step_outputs returns a dictionary with the following
        key-value pairs:

            * 'representations': (N, *) array where N is the number of structures
            * 'representations_energy': (N,) array where N is the number of structures
        """
        # On each worker, compute the per-structure average representations
        per_struct_representations  = []
        per_struct_energies         = []
        for s in step_outputs:
            per_struct_representations.append(s['representations'])
            per_struct_energies.append(s['representations_energy'])

        per_struct_representations = torch.vstack(per_struct_representations)
        per_struct_energies        = torch.cat(per_struct_energies)

        # Now gather everything
        per_struct_representations = self.all_gather(per_struct_representations)
        per_struct_energies = self.all_gather(per_struct_energies)

        # Reshape to remove the num_processes dimension.
        # NOTE: order is likely not going to match dataloader order
        per_struct_representations = torch.flatten(
            per_struct_representations, 0, 1
        )
        per_struct_energies = torch.flatten(per_struct_energies, 0, 1)

        n_reps = per_struct_representations.shape[0]
        n_engs = per_struct_energies.shape[0]

        assert n_reps == n_engs, "Incompatible shapes: {} representations, {} atoms".format(n_reps, n_engs)

        self.results['representations'] = per_struct_representations
        self.results['representations_energies'] = per_struct_energies


    def aggregate_atom_representations(self, step_outputs):
        # On each worker, compute the per-structure average representations
        per_atom_representations  = []
        per_atom_energies         = []
        per_struct_splits         = []
        for s in step_outputs:
            per_atom_representations.append(s['representations'])
            per_atom_energies.append(s['representations_energy'])
            per_struct_splits.append(torch.Tensor(s['representations_splits']))

        per_atom_representations = torch.vstack(per_atom_representations)
        per_atom_energies        = torch.cat(per_atom_energies)
        per_struct_splits        = torch.cat(per_struct_splits)

        # Now gather everything
        per_atom_representations = self.all_gather(per_atom_representations)
        per_atom_energies = self.all_gather(per_atom_energies)
        per_struct_splits = self.all_gather(per_struct_splits)

        # Reshape to remove the num_processes dimension.
        # NOTE: order is likely not going to match dataloader order
        per_atom_representations = torch.flatten(
            per_atom_representations, 0, 1
        )
        per_atom_energies = torch.flatten(per_atom_energies, 0, 1)
        per_struct_splits = torch.flatten(per_struct_splits, 0, 1)

        n_reps = per_atom_representations.shape[0]
        n_engs = per_atom_energies.shape[0]

        assert n_reps == n_engs, "Incompatible shapes: {} representations, {} atoms".format(n_reps, n_engs)
        assert per_struct_splits.sum() == n_reps, "Incompatible shapes: {} representations, {} splits sum".format(n_reps, per_struct_splits.sum())

        self.results['representations'] = per_atom_representations
        self.results['representations_energies'] = per_atom_energies
        self.results['representations_splits'] = per_struct_splits


    def on_test_epoch_start(self):
        if self.reset_results_on_epoch_start:
            self.results = {}


    def test_step(self, batch, batch_idx):
        torch.set_grad_enabled(True)

        results = {}
        for value in self.values_to_compute:
            fxn = getattr(self, f'compute_{value}')

            for k,v in fxn(batch).items():
                if k in results:
                    raise RuntimeError(f"Key '{k}' cannot be returned by multiple compute functions")

                results[k] = v


        for k,v in results.items():
            if isinstance(v, torch.Tensor):
                results[k] = v.detach()#.cpu().numpy()

        return results

    def test_epoch_end(self, step_outputs):
        for value in self.values_to_compute:
            aggregation_fxn = getattr(self, f'aggregate_{value}')
            aggregation_fxn(step_outputs)


    def copy(self, model_path):
        """
        Copies all files necessary for model construction to the current
        working directory.

        Args:

            model_path (str):
                The path to a folder containing the information necessary to
                load the model.

        """
        raise NotImplementedError
