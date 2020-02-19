#!/usr/bin/env python3

import sys
import argparse

import torch
import torch.nn as nn
import torch.distributed as dist

import deepspeed


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
    def __init__(self, engine_base, engine_test, loader):
        self.base_eng = engine_base
        self.test_eng = engine_test
        self.loader = loader

        self.print_checks = True

        self.track_params = True
        self.track_loss = True
        self.track_grads = True

        self.global_rank = dist.get_rank()
        self.device = self.base_eng.device

        self.base_loss = -1.
        self.test_loss = -1.

        assert self.base_eng.global_rank == self.test_eng.global_rank

    def step_batch(self, inputs, labels):
        # Prepare identical baseline and test data
        base_inputs = inputs.clone().to(self.device)
        base_labels = labels.clone().to(self.device)

        test_inputs = inputs.clone().to(self.device)
        test_labels = labels.clone().to(self.device)

        # Take a baseline step
        self.base_eng.optimizer.zero_grad()
        self.base_loss = self.base_eng.module(base_inputs, base_labels)
        self.base_eng.backward(self.base_loss)
        self.base_eng.optimizer.step()

        # Take a test step
        self.test_eng.optimizer.zero_grad()
        self.test_loss = self.test_eng.module(test_inputs, test_labels)
        self.test_eng.backward(self.test_loss)
        self.test_eng.optimizer.step()

    def get_params(self):
        # XXX TODO: there should be a gather here if eng.mpu is not None
        base_params = [
            p for p in self.base_eng.module.parameters() if p.requires_grad
        ]
        test_params = [
            p for p in self.test_eng.module.parameters() if p.requires_grad
        ]
        assert len(base_params) == len(test_params)
        return base_params, test_params

    def get_grads(self):
        # XXX TODO: there should be a gather here if eng.mpu is not None
        base_grads = [
            p for p in self.base_eng.module.parameters() if p.requires_grad
        ]
        test_grads = [
            p for p in self.test_eng.module.parameters() if p.requires_grad
        ]
        assert len(base_grads) == len(test_grads)
        return base_grads, test_grads

    def go(self,
           num_batches=10,
           test_every=1,
           status_every=1,
           param_tol=1e-2,
           loss_tol=1e-4,
           grad_tol=1e-2):
        for batch_idx, data in enumerate(self.loader):
            if batch_idx == num_batches:
                break
            inputs = data[0]
            labels = data[1]

            self.step_batch(inputs, labels)

            if batch_idx % status_every == 0:
                loss_t = torch.Tensor([self.base_loss,
                                       self.test_loss]).to(self.device)
                dist.all_reduce(loss_t)
                loss_t = loss_t / dist.get_world_size()
                abs_ = abs_diff(loss_t[0], loss_t[1])
                rel_ = rel_diff(loss_t[0], loss_t[1])
                if self.global_rank == 0:
                    print(
                        f'batch={batch_idx} / {num_batches} '
                        f'base_loss={loss_t[0]:0.5f} test_loss={loss_t[1]:0.5f} '
                        f'abs_diff={abs_:0.5e} rel_diff={rel_:0.5e}')

            # Move on if we're not checking results
            if batch_idx % test_every != 0:
                continue

            test_pass = True
            with torch.no_grad():
                if self.track_params:
                    base_params, test_params = self.get_params()
                    for p_idx in range(len(base_params)):
                        abs_ = abs_diff(base_params[p_idx], test_params[p_idx])
                        rel_ = rel_diff(base_params[p_idx], test_params[p_idx])
                        if rel_ > param_tol:
                            print(
                                f'ERROR rank={self.global_rank} batch={batch_idx}: '
                                f'PARAMETER divergence for param={p_idx} '
                                f'abs_diff={abs_:0.5e} rel_diff={rel_:0.5e} '
                                f'tol={param_tol:0.5e}')
                            test_pass = False

                if self.track_loss:
                    abs_ = abs_diff(self.base_loss, self.test_loss)
                    rel_ = rel_diff(self.base_loss, self.test_loss)
                    if rel_ > loss_tol:
                        print(
                            f'ERROR rank={self.global_rank} batch={batch_idx}: '
                            f'LOSS divergence '
                            f'base={self.base_loss:0.5e} test={self.test_loss:0.5e} '
                            f'abs_diff={abs_:0.5f} rel_diff={rel_:0.5e} '
                            f'tol={loss_tol:0.5e}')
                        test_pass = False

                if self.track_grads:
                    base_grads, test_grads = self.get_grads()
                    assert len(base_grads) == len(test_grads)
                    for g_idx in range(len(base_grads)):
                        abs_ = abs_diff(base_grads[g_idx], test_grads[g_idx])
                        rel_ = rel_diff(base_grads[g_idx], test_grads[g_idx])
                        if rel_ > grad_tol:
                            print(
                                f'ERROR rank={self.global_rank} batch={batch_idx}: '
                                f'GRADIENT divergence for param={g_idx} '
                                f'abs_diff={abs_:0.5e} rel_diff={rel_:0.5e} '
                                f'tol={grad_tol:0.5e}')
                            test_pass = False

            status_t = torch.Tensor([test_pass == False]).to(self.device)
            dist.all_reduce(status_t)
            if status_t[0] > 0:
                return False

        return True


def cifar_loader(batch_size):
    """Construct DataLoader for CIFAR10 train data. """
    import torchvision
    import torchvision.transforms as transforms

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data',
                                            train=True,
                                            download=True,
                                            transform=transform)
    sampler = torch.utils.data.distributed.DistributedSampler(trainset,
                                                              shuffle=False)

    trainloader = torch.utils.data.DataLoader(trainset,
                                              sampler=sampler,
                                              batch_size=batch_size,
                                              pin_memory=True)
    return trainloader


def get_cmd_args():
    parser = argparse.ArgumentParser(description='CIFAR')
    parser.add_argument('--local_rank',
                        type=int,
                        default=-1,
                        help='local rank passed from distributed launcher')
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_cmd_args()

    from SimpleNet import SimpleNet

    torch.manual_seed(1138)
    base_net = SimpleNet()
    base_opt = torch.optim.Adam(base_net.parameters(), betas=(0.9, 0.999))

    base_engine, base_opt, __, __ = deepspeed.initialize(
        args=args,
        model=base_net,
        optimizer=base_opt,
        model_parameters=base_net.parameters(),
        dist_init_required=True)

    torch.manual_seed(1138)
    test_net = SimpleNet()
    test_net.load_state_dict(base_net.state_dict())  # copy from base
    test_opt = torch.optim.Adam(test_net.parameters(), betas=(0.9, 0.999))
    #test_opt = torch.optim.Adam(test_net.parameters(), betas=(0.900001, 0.999))

    test_engine, test_opt, __, __ = deepspeed.initialize(
        args=args,
        model=test_net,
        optimizer=test_opt,
        model_parameters=test_net.parameters(),
        dist_init_required=False)

    trainloader = cifar_loader(base_engine.train_micro_batch_size_per_gpu())

    stepper = ModelStepper(base_engine, test_engine, trainloader)

    correct = stepper.go(num_batches=1000, test_every=100, status_every=50)

    if correct:
        print('TEST PASSED')
    else:
        print('TEST FAILED')
