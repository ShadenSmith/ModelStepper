#!/usr/bin/env python3

import torch
import torch.distributed as dist


def abs_diff(A, B):
    """ Compute the norm of the difference between A and B. """
    if type(A) == float:
        return abs(b - a)
    else:
        # Try some Tensor-derived type
        return (B - A).norm()


def rel_diff(A, B):
    """ Compute the relative (percent) difference between A and B. """
    if type(A) == float:
        return abs((B - A) / A)
    else:
        # Try some Tensor-derived type
        return ((B - A).norm() / A.norm())


class ModelStepper:
    def __init__(self,
                 engine_base,
                 engine_test,
                 loader,
                 num_batches=10,
                 test_every=1,
                 status_every=1,
                 track_params=True,
                 track_loss=True,
                 track_grads=True,
                 param_tol=1e-4,
                 loss_tol=1e-4,
                 grad_tol=1e-4,
                 batch_data_fn=None):
        """A correctness checker for DeepSpeed.

        Input:
            engine_base: the baseline returned from deepspeed.initialize()
            engine_test: the tested engine returned from deepspeed.initialize()
            loader (DataLoader): ModelStepper will clone input for you.

        """

        self.base_eng = engine_base
        self.test_eng = engine_test
        self.loader = loader

        self.print_checks = True

        self.global_rank = dist.get_rank()
        self.device = self.base_eng.device

        self.base_loss = -1.
        self.test_loss = -1.

        self.num_batches = num_batches
        self.test_every = test_every
        self.status_every = status_every
        self.track_params = track_params
        self.track_loss = track_loss
        self.track_grads = track_grads
        self.param_tol = param_tol
        self.loss_tol = loss_tol
        self.grad_tol = grad_tol
        self.batch_data_fn = batch_data_fn

        assert self.base_eng.global_rank == self.test_eng.global_rank

        if dist.get_rank() == 0:
            print()
            print('--- Model Stepper Configuration ---')
            print(f'batches={num_batches}\n'
                  f'test_every={test_every}\nstatus_every={status_every}\n'
                  f'track_params={track_params}\nparam_tol={param_tol:e}\n'
                  f'track_loss={track_loss}\nloss_tol={loss_tol:e}\n'
                  f'track_grads={track_grads}\ngrad_tol={grad_tol:e}')
            print()

    def _get_params(self):
        # TODO; if models do not use identical model parallelism, a gather should be used
        base_params = [p for p in self.base_eng.module.parameters() if p.requires_grad]
        test_params = [p for p in self.test_eng.module.parameters() if p.requires_grad]
        assert len(base_params) == len(test_params)
        return base_params, test_params

    def _get_grads(self):
        # TODO; if models do not use identical model parallelism, a gather should be used
        base_grads = [p for p in self.base_eng.module.parameters() if p.requires_grad]
        test_grads = [p for p in self.test_eng.module.parameters() if p.requires_grad]
        assert len(base_grads) == len(test_grads)
        return base_grads, test_grads

    def _step_batch(self, batch_idx, base_batch_data, test_batch_data):
        self.base_eng.optimizer.zero_grad()
        self.test_eng.optimizer.zero_grad()

        # Should we test parameters/loss/gradients this batch?
        test_batch = (batch_idx %
                      self.test_every == 0) or (batch_idx == self.num_batches - 1)

        # Forward pass
        self.base_loss = self.base_eng.module(*base_batch_data)
        self.test_loss = self.test_eng.module(*test_batch_data)

        if test_batch and self.track_loss:
            abs_ = abs_diff(self.base_loss, self.test_loss)
            rel_ = rel_diff(self.base_loss, self.test_loss)
            if rel_ > self.loss_tol:
                print(f'DIVERGED LOSS rank={self.global_rank} batch={batch_idx}: '
                      f'base={self.base_loss:0.5e} test={self.test_loss:0.5e} '
                      f'abs_diff={abs_:0.5f} rel_diff={rel_:0.5e} '
                      f'tol={self.loss_tol:0.5e}')
                return False

        # Backward pass
        self.base_eng.backward(self.base_loss)
        self.test_eng.backward(self.test_loss)

        if test_batch and self.track_grads:
            with torch.no_grad():
                base_grads, test_grads = self._get_grads()
                assert len(base_grads) == len(test_grads)
                for g_idx in range(len(base_grads)):
                    abs_ = abs_diff(base_grads[g_idx], test_grads[g_idx])
                    rel_ = rel_diff(base_grads[g_idx], test_grads[g_idx])
                    if rel_ > self.grad_tol:
                        print(f'DIVERGED GRADIENT rank={self.global_rank} '
                              f'batch={batch_idx} grad_idx={g_idx} '
                              f'abs_diff={abs_:0.5e} rel_diff={rel_:0.5e} '
                              f'tol={self.grad_tol:0.5e}')
                        test_pass = False

        # Update the models
        self.base_eng.optimizer.step()
        self.test_eng.optimizer.step()

        if test_batch and self.track_params:
            with torch.no_grad():
                base_params, test_params = self._get_params()
                for p_idx in range(len(base_params)):
                    abs_ = abs_diff(base_params[p_idx], test_params[p_idx])
                    rel_ = rel_diff(base_params[p_idx], test_params[p_idx])
                    if rel_ > self.param_tol:
                        print(f'DIVERGED PARAMETER rank={self.global_rank} '
                              f'batch={batch_idx} param_idx={p_idx} '
                              f'abs_diff={abs_:0.5e} rel_diff={rel_:0.5e} '
                              f'tol={self.param_tol:0.5e}')
                        return False

        return True

    def _print_status(self, batch_idx):
        loss_t = torch.Tensor([self.base_loss, self.test_loss]).to(self.device)
        dist.all_reduce(loss_t)
        loss_t = loss_t / dist.get_world_size()
        abs_ = abs_diff(loss_t[0], loss_t[1])
        rel_ = rel_diff(loss_t[0], loss_t[1])
        if self.global_rank == 0:
            print(f'STATUS batch={batch_idx} / {self.num_batches} '
                  f'base_loss={loss_t[0]:0.5f} test_loss={loss_t[1]:0.5f} '
                  f'abs_diff={abs_:0.5e} rel_diff={rel_:0.5e}')

    def go(self):
        for batch_idx, data in enumerate(self.loader):
            if batch_idx == self.num_batches:
                self._print_status(batch_idx)
                break

            # Prepare identical baseline and test data
            if self.batch_data_fn:
                base_data = self.batch_data_fn(data)
                test_data = self.batch_data_fn(data)
            else:
                # Assume something easy
                if type(data) == list:
                    base_data = [x.clone().to(self.base_eng.device) for x in data]
                    test_data = [x.clone().to(self.test_eng.device) for x in data]
                else:
                    base_data = data.clone().to(self.base_eng.device)
                    test_data = data.clone().to(self.test_eng.device)

            test_pass = self._step_batch(batch_idx, base_data, test_data)

            # See if anyone failed.
            status_t = torch.Tensor([test_pass == False]).to(self.device)
            dist.all_reduce(status_t)
            if status_t[0] > 0:
                self._print_status(batch_idx)
                if self.global_rank == 0:
                    print('TEST FAILED')
                return False

            if batch_idx % self.status_every == 0:
                self._print_status(batch_idx)

        if self.global_rank == 0:
            print('TEST PASSED')
        return True
