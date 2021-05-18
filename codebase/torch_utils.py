"""
Elliot Schumacher, Johns Hopkins University
Created 2/12/19
From torch_utils in spotlight project (see https://github.com/maciejkula/spotlight)
"""

import numpy as np

import torch
import random
import logging

log = logging.getLogger()

def gpu(tensor, gpu=False):

    if gpu:
        return tensor.cuda()
    else:
        return tensor


def cpu(tensor):

    if tensor.is_cuda:
        return tensor.cpu()
    else:
        return tensor


def minibatch(*tensors, **kwargs):

    batch_size = kwargs.get('batch_size', 128)

    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i + batch_size] for x in tensors)


def shuffle(*arrays, **kwargs):

    random_state = kwargs.get('random_state')

    if len(set(len(x) for x in arrays)) != 1:
        raise ValueError('All inputs to shuffle must have '
                         'the same length.')

    if random_state is None:
        random_state = np.random.RandomState()

    shuffle_indices = np.arange(len(arrays[0]))
    random_state.shuffle(shuffle_indices)

    if len(arrays) == 1:
        return arrays[0][shuffle_indices]
    else:
        return tuple(x[shuffle_indices] for x in arrays)


def assert_no_grad(variable):

    if variable.requires_grad:
        raise ValueError(
            "nn criterions don't compute the gradient w.r.t. targets - please "
            "mark these variables as volatile or not requiring gradients"
        )


def set_seed(seed=None, cuda=False):

    if seed is None:
        seed = random.randint(0, 2**32 - 1)

    log.info("Setting seed to be {0}".format(seed))
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if cuda:
        torch.cuda.manual_seed(seed)


def main():
    pass


if __name__ == "__main__":
    main()