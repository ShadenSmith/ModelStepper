import argparse
import torch

import deepspeed

from ModelStepper import ModelStepper


def cifar_loader(batch_size):
    """Construct DataLoader for CIFAR10 train data. """
    import torchvision
    import torchvision.transforms as transforms

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5,
                               0.5,
                               0.5),
                              (0.5,
                               0.5,
                               0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data',
                                            train=True,
                                            download=True,
                                            transform=transform)
    sampler = torch.utils.data.distributed.DistributedSampler(trainset, shuffle=False)

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
    base_opt = torch.optim.Adam(base_net.parameters())

    base_engine, __, __, __ = deepspeed.initialize(
        args=args,
        model=base_net,
        optimizer=base_opt,
        model_parameters=base_net.parameters(),
        dist_init_required=True)

    torch.manual_seed(1138)
    test_net = SimpleNet()
    test_net.load_state_dict(base_net.state_dict())  # copy from base
    test_opt = torch.optim.Adam(test_net.parameters())

    # uncomment this to fail
    #test_opt = torch.optim.Adam(test_net.parameters(), lr=0.)

    test_engine, __, __, __ = deepspeed.initialize(
        args=args,
        model=test_net,
        optimizer=test_opt,
        model_parameters=test_net.parameters(),
        dist_init_required=False)

    trainloader = cifar_loader(base_engine.train_micro_batch_size_per_gpu())

    stepper = ModelStepper(base_engine,
                           test_engine,
                           trainloader,
                           num_batches=50,
                           test_every=1,
                           status_every=5,
                           loss_tol=1e-5,
                           track_grads=True,
                           track_params=True)

    correct = stepper.go()
    if correct:
        print('TEST PASSED')
    else:
        print('TEST FAILED')
