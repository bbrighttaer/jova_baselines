# Author: bbrighttaer
# Project: jova
# Date: 5/24/19
# Time: 12:27 AM
# File: math.py


from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import OrderedDict, namedtuple

import torch


def segment_sum(data, segment_ids):
    """
    Analogous to tf.segment_sum (https://www.tensorflow.org/api_docs/python/tf/math/segment_sum).

    :param data: A pytorch tensor of the data for segmented summation.
    :param segment_ids: A 1-D tensor containing the indices for the segmentation.
    :return: a tensor of the same type as data containing the results of the segmented summation.
    """
    if not all(segment_ids[i] <= segment_ids[i + 1] for i in range(len(segment_ids) - 1)):
        raise AssertionError("elements of segment_ids must be sorted")

    if len(segment_ids.shape) != 1:
        raise AssertionError("segment_ids have be a 1-D tensor")

    if data.shape[0] != segment_ids.shape[0]:
        raise AssertionError("segment_ids should be the same size as dimension 0 of input.")

    # t_grp = {}
    # idx = 0
    # for i, s_id in enumerate(segment_ids):
    #     s_id = s_id.item()
    #     if s_id in t_grp:
    #         t_grp[s_id] = t_grp[s_id] + data[idx]
    #     else:
    #         t_grp[s_id] = data[idx]
    #     idx = i + 1
    #
    # lst = list(t_grp.values())
    # tensor = torch.stack(lst)

    num_segments = len(torch.unique(segment_ids))
    return unsorted_segment_sum(data, segment_ids, num_segments)


def cuda(tensor):
    from jova import cuda
    if cuda:
        return tensor.cuda()
    else:
        return tensor


def unsorted_segment_sum(data, segment_ids, num_segments):
    """
    Computes the sum along segments of a tensor. Analogous to tf.unsorted_segment_sum.

    :param data: A tensor whose segments are to be summed.
    :param segment_ids: The segment indices tensor.
    :param num_segments: The number of segments.
    :return: A tensor of same data type as the data argument.
    """
    assert all([i in data.shape for i in segment_ids.shape]), "segment_ids.shape should be a prefix of data.shape"

    # segment_ids is a 1-D tensor repeat it to have the same shape as data
    if len(segment_ids.shape) == 1:
        s = torch.prod(torch.tensor(data.shape[1:])).long()
        s = cuda(s)
        segment_ids = segment_ids.repeat_interleave(s).view(segment_ids.shape[0], *data.shape[1:])

    assert data.shape == segment_ids.shape, "data.shape and segment_ids.shape should be equal"

    shape = [num_segments] + list(data.shape[1:])
    tensor = cuda(torch.zeros(*shape)).scatter_add(0, segment_ids.long(), data.float())
    tensor = tensor.type(data.dtype)
    return tensor


def unsorted_segment_max(data, segment_ids, num_segments):
    # TODO(bbrighttaer): Optimize this function
    """
    Computes the sum along segments of a tensor. Analogous to tf.unsorted_segment_max.

    :param data: A tensor whose segments are to be summed.
    :param segment_ids: The segment indices tensor.
    :param num_segments: The number of segments.
    :return: A tensor of same data type as the data argument.
    """
    assert all([i in data.shape for i in segment_ids.shape]), "segment_ids.shape should be a prefix of data.shape"

    t_grp = OrderedDict()
    idx = 0
    for i, s_id in enumerate(segment_ids):
        s_id = s_id.item()
        if s_id in t_grp:
            t_grp[s_id] = torch.max(t_grp[s_id], data[idx])
        else:
            t_grp[s_id] = data[idx]
        idx = i + 1

    lst = list(t_grp.values())
    tensor = torch.stack(lst)
    return tensor


# def unsorted_segment_sum(data, segment_ids, num_segments):
#     """
#     Computes the sum along segments of a tensor. Analogous to tf.unsorted_segment_sum.
#
#     :param data: A tensor whose segments are to be summed.
#     :param segment_ids: The segment indices tensor.
#     :param num_segments: The number of segments.
#     :return: A tensor of same data type as the data argument.
#     """
#     if len(data.shape) == 1:
#         data = torch.unsqueeze(data.squeeze(), dim=0)
#     if len(segment_ids.shape) == 1:
#         segment_ids = torch.unsqueeze(segment_ids.squeeze(), dim=0)
#     shape = list(segment_ids.shape[:-1]) + [num_segments]
#     zero_tensor = torch.zeros(shape)
#     tensor = zero_tensor.scatter_add(1, segment_ids, data)
#     return tensor

# def segment_sum(data, segment_ids):
#     """
#     Analogous to tf.segment_sum (https://www.tensorflow.org/api_docs/python/tf/math/segment_sum).
#
#     :param data: A pytorch tensor of the data for segmented summation.
#     :param segment_ids: A 1-D tensor containing the indices for the segmentation.
#     :return: a tensor of the same type as data containing the results of the segmented summation.
#     """
#     try:
#         assert data.shape[0] == segment_ids.shape[0]
#     except AssertionError:
#         logger = get_logger(level='error')
#         logger.error("segment_ids should be the same size as dimension 0 of input.")
#
#     grp = {}
#     for i, val in enumerate(data):
#         idx = segment_ids[i].item()
#         val = torch.sum(val)
#         if idx in grp:
#             grp[idx] = grp[idx] + val
#         else:
#             grp[idx] = val
#     rows = list(grp.values())
#     tensor = torch.tensor(rows, dtype=data.dtype)
#     return tensor

class ExpAverage(object):
    def __init__(self, beta, bias_cor=False):
        self.beta = beta
        self.value = 0.
        self.bias_cor = bias_cor
        self.t = 0

    def reset(self):
        self.t = 0
        self.value = 0

    def update(self, v):
        self.t += 1
        self.value = self.beta * self.value + (1. - self.beta) * v
        if self.bias_cor:
            self.value = self.value / (1. - pow(self.beta, self.t))


class Count(object):
    def __init__(self, i=-1):
        self.i = i

    def inc(self):
        self.i += 1

    def getAndInc(self):
        r = self.i
        self.inc()
        return r

    def IncAndGet(self):
        self.inc()
        return self.i


######################################################################################
"""
The functions within this block are from: https://github.com/yulkang/pylabyk/blob/master/numpytorch.py
"""


def permute2st(v, ndim_en=1):
    """
    Permute last ndim_en of tensor v to the first
    :type v: torch.Tensor
    :type ndim_en: int
    :rtype: torch.Tensor
    """
    nd = v.ndimension()
    return v.permute([*range(-ndim_en, 0)] + [*range(nd - ndim_en)])


def permute2en(v, ndim_st=1):
    """
    Permute last ndim_en of tensor v to the first
    :type v: torch.Tensor
    :type ndim_st: int
    :rtype: torch.Tensor
    """
    nd = v.ndimension()
    return v.permute([*range(ndim_st, nd)] + [*range(ndim_st)])


def block_diag(m):
    """
    Make a block diagonal matrix along dim=-3
    EXAMPLE:
    block_diag(torch.ones(4,3,2))
    should give a 12 x 8 matrix with blocks of 3 x 2 ones.
    Prepend batch dimensions if needed.
    You can also give a list of matrices.
    :type m: torch.Tensor, list
    :rtype: torch.Tensor
    """
    if type(m) is list:
        m = torch.cat([m1.unsqueeze(-3) for m1 in m], -3)

    d = m.dim()
    n = m.shape[-3]
    siz0 = m.shape[:-3]
    siz1 = m.shape[-2:]
    m2 = m.unsqueeze(-2)
    eye = attach_dim(torch.eye(n).unsqueeze(-2), d - 3, 1)
    return (m2 * eye).reshape(
        siz0 + torch.Size(torch.tensor(siz1) * n)
    )


def attach_dim(v, n_dim_to_prepend=0, n_dim_to_append=0):
    return v.reshape(
        torch.Size([1] * n_dim_to_prepend)
        + v.shape
        + torch.Size([1] * n_dim_to_append))


def block_diag_irregular(matrices):
    # Block diagonal from a list of matrices that have different shapes.
    # If they have identical shapes, use block_diag(), which is vectorized.

    matrices = [permute2st(m, 2) for m in matrices]

    ns = torch.LongTensor([m.shape[0] for m in matrices])
    n = torch.sum(ns)
    batch_shape = matrices[0].shape[2:]

    v = torch.zeros(torch.Size([n, n]) + batch_shape)
    for ii, m1 in enumerate(matrices):
        st = torch.sum(ns[:ii])
        en = torch.sum(ns[:(ii + 1)])
        v[st:en, st:en] = m1
    return permute2en(v, 2)

######################################################################################
