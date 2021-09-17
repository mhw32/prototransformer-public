import os
import json
import shutil
import torch
import scipy
import inspect
import numpy as np
from scipy.stats import t
from collections import Counter, OrderedDict


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def copy_checkpoint(folder='./', filename='checkpoint.pth.tar',
                    copyname='copy.pth.tar'):
    shutil.copyfile(os.path.join(folder, filename),
                    os.path.join(folder, copyname))


def save_checkpoint(state, is_best, folder='./', filename='checkpoint.pth.tar'):
    if not os.path.isdir(folder):
        os.mkdir(folder)
    torch.save(state, os.path.join(folder, filename))
    if is_best:
        shutil.copyfile(os.path.join(folder, filename),
                        os.path.join(folder, 'model_best.pth.tar'))


def load_json(f_path):
    with open(f_path, 'r') as f:
        return json.load(f)


def save_json(obj, f_path):
    with open(f_path, 'w') as f:
        json.dump(obj, f, ensure_ascii=False)


class OrderedCounter(Counter, OrderedDict):
    """Counter that remembers the order elements are first encountered"""

    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__, OrderedDict(self))

    def __reduce__(self):
        return self.__class__, (OrderedDict(self),)


def l2_normalize(x, dim=1):
    return x / torch.sqrt(torch.sum(x**2, dim=dim).unsqueeze(dim))


def wlogsumexp(inputs, weights, dim, keepdim=False):
    inputs_max = torch.max(inputs, dim=dim, keepdim=True)
    inputs_dif = inputs - inputs_max
    sum_of_exp = torch.sum(weights * torch.exp(inputs_dif))
    return inputs_max + torch.log(sum_of_exp)


def product_of_experts(mu_experts, var_experts, eps=1e-7):
    """
    Return parameters for product of independent experts.
    See https://arxiv.org/pdf/1410.7827.pdf for equations.
    
    @param mu_experts  : batch_size x num_experts x latent_dim
    @param var_experts : batch_size x num_experts x latent_dim
    """
    T = 1. / (var_experts + eps)
    mu_product = torch.sum(mu_experts * T, dim=1) / torch.sum(T, dim=1)
    var_product = 1. / torch.sum(T, dim=1)
    return mu_product, var_product + eps


def exclude_bn_weight_bias_from_weight_decay(model, weight_decay):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if 'bn' in name:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}
    ]


def string_concat(arr1, arr2):
    list1 = arr1.tolist()
    list2 = arr2.tolist()
    return np.array([f'{l1}-{l2}' for l1, l2 in zip(list1, list2)])


def adjust_learning_rate(
        optimizer,
        epoch,
        learning_rate,
        lr_decay_epochs,
        lr_decay_rate,
    ):
    """Sets the learning rate to the initial LR decayed by decay rate every steep step"""
    iterations = lr_decay_epochs.split(',')
    lr_decay_epochs = []
    for it in iterations:
        lr_decay_epochs.append(int(it))
    steps = np.sum(epoch > np.asarray(lr_decay_epochs))
    if steps > 0:
        new_lr = learning_rate * (lr_decay_rate ** steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr


def get_accuracy(output, target):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    # output: batch_size x n*m x n
    # target: batch_size x n*m

    with torch.no_grad():
        batch_size = target.size(0)
        pred = torch.argmax(output, dim=2)
        correct = pred.eq(target).sum(1)
        acc = 100. / target.size(1) * correct
        return acc.cpu().numpy()


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * t._ppf((1+confidence)/2., n-1)
    return m, h


def reset_model_for_training(model):
    model.train()
    for param in model.parameters():
        param.requires_grad = True


def reset_model_for_evaluation(model):
    model.eval()
    for param in model.parameters():
        param.requires_grad = False


def my_apply_chunking_to_forward(forward_fn, chunk_size, chunk_dim, *input_tensors):
    assert len(input_tensors) > 0, "{} has to be a tuple/list of tensors".format(input_tensors)
    
    # inspect.signature exist since python 3.5 and is a python method -> no problem with backward compability
    num_args_in_forward_chunk_fn = len(inspect.signature(forward_fn).parameters)
    assert num_args_in_forward_chunk_fn == len(
        input_tensors
    ), "forward_chunk_fn expects {} arguments, but only {} input tensors are given".format(
        num_args_in_forward_chunk_fn, len(input_tensors)
    )

    if chunk_size > 0:
        assert (
            input_tensors[0].shape[chunk_dim] % chunk_size == 0
        ), "The dimension to be chunked {} has to be a multiple of the chunk size {}".format(
            input_tensors[0].shape[chunk_dim], chunk_size
        )

        num_chunks = input_tensors[0].shape[chunk_dim] // chunk_size

        # chunk input tensor into tuples
        input_tensors_chunks = tuple(input_tensor.chunk(num_chunks, dim=chunk_dim) for input_tensor in input_tensors)
        # apply forward fn to every tuple
        output_chunks = tuple(forward_fn(*input_tensors_chunk) for input_tensors_chunk in zip(*input_tensors_chunks))
        # concatenate output at same dimension
        return torch.cat(output_chunks, dim=chunk_dim)

    return forward_fn(*input_tensors)
