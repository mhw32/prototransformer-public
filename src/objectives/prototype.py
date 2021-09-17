import torch


def euclidean_dist(x, y):
    '''
    Compute euclidean distance between two tensors
    '''
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception

    x = x.unsqueeze(1)
    y = y.unsqueeze(0)

    return torch.pow(x - y, 2).sum(2)


def batch_euclidean_dist(x, y):
    # x: B x N x D
    # y: B x M x D
    b, n, d = x.size()
    m = y.size(1)

    # x : B x N x 1 x D
    # y : B x 1 x M x D
    x = x.unsqueeze(2)
    y = y.unsqueeze(1)

    # dists : B x N x M
    dists = torch.pow(x - y, 2).sum(3)
    return dists

