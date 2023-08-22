# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/util/misc.py
"""
Misc functions, including distributed helpers.

Mostly copy-paste from torchvision references.
"""
from .log import logger

from typing import List, Optional
from collections import OrderedDict
from scipy.io import loadmat
import numpy as np
import csv
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.distributed as dist
import torchvision
from torch import Tensor


def _max_by_axis(the_list):
    # type: (List[List[int]]) -> List[int]
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


def get_world_size() -> int:
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        # type: (Device) -> NestedTensor # noqa
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)


def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
    # TODO make this more general
    if tensor_list[0].ndim == 3:
        if torchvision._is_tracing():
            # nested_tensor_from_tensor_list() does not export well to ONNX
            # call _onnx_nested_tensor_from_tensor_list() instead
            return _onnx_nested_tensor_from_tensor_list(tensor_list)

        # TODO make it support different-sized images
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], : img.shape[2]] = False
    else:
        raise ValueError("not supported")
    return NestedTensor(tensor, mask)


# _onnx_nested_tensor_from_tensor_list() is an implementation of
# nested_tensor_from_tensor_list() that is supported by ONNX tracing.
@torch.jit.unused
def _onnx_nested_tensor_from_tensor_list(tensor_list: List[Tensor]) -> NestedTensor:
    max_size = []
    for i in range(tensor_list[0].dim()):
        max_size_i = torch.max(
            torch.stack([img.shape[i] for img in tensor_list]).to(torch.float32)
        ).to(torch.int64)
        max_size.append(max_size_i)
    max_size = tuple(max_size)

    # work around for
    # pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
    # m[: img.shape[1], :img.shape[2]] = False
    # which is not yet supported in onnx
    padded_imgs = []
    padded_masks = []
    for img in tensor_list:
        padding = [(s1 - s2) for s1, s2 in zip(max_size, tuple(img.shape))]
        padded_img = torch.nn.functional.pad(img, (0, padding[2], 0, padding[1], 0, padding[0]))
        padded_imgs.append(padded_img)

        m = torch.zeros_like(img[0], dtype=torch.int, device=img.device)
        padded_mask = torch.nn.functional.pad(m, (0, padding[2], 0, padding[1]), "constant", 1)
        padded_masks.append(padded_mask.to(torch.bool))

    tensor = torch.stack(padded_imgs)
    mask = torch.stack(padded_masks)

    return NestedTensor(tensor, mask=mask)


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def load_parallal_model(model, state_dict_):
    state_dict = OrderedDict()
    for key in state_dict_:
        if key.startswith('module') and not key.startswith('module_list'):
            state_dict[key[7:]] = state_dict_[key]
        else:
            state_dict[key] = state_dict_[key]

    # check loaded parameters and created model parameters
    model_state_dict = model.state_dict()
    for key in state_dict:
        if key in model_state_dict:
            if state_dict[key].shape != model_state_dict[key].shape:
                print('Skip loading parameter {}, required shape{}, loaded shape{}.'.format(
                    key, model_state_dict[key].shape, state_dict[key].shape))
                state_dict[key] = model_state_dict[key]
        else:
            print('Drop parameter {}.'.format(key))
    for key in model_state_dict:
        if key not in state_dict:
            print('No param {}.'.format(key))
            state_dict[key] = model_state_dict[key]
    model.load_state_dict(state_dict, strict=False)

    return model


class ADEVisualize(object):
    def __init__(self):
        self.colors = loadmat('dataset/color150.mat')['colors']
        self.names = {}
        with open('dataset/object150_info.csv') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                self.names[int(row[0])] = row[5].split(";")[0]

    def unique(self, ar, return_index=False, return_inverse=False, return_counts=False):
        ar = np.asanyarray(ar).flatten()

        optional_indices = return_index or return_inverse
        optional_returns = optional_indices or return_counts

        if ar.size == 0:
            if not optional_returns:
                ret = ar
            else:
                ret = (ar,)
                if return_index:
                    ret += (np.empty(0, np.bool),)
                if return_inverse:
                    ret += (np.empty(0, np.bool),)
                if return_counts:
                    ret += (np.empty(0, np.intp),)
            return ret
        if optional_indices:
            perm = ar.argsort(kind='mergesort' if return_index else 'quicksort')
            aux = ar[perm]
        else:
            ar.sort()
            aux = ar
        flag = np.concatenate(([True], aux[1:] != aux[:-1]))

        if not optional_returns:
            ret = aux[flag]
        else:
            ret = (aux[flag],)
            if return_index:
                ret += (perm[flag],)
            if return_inverse:
                iflag = np.cumsum(flag) - 1
                inv_idx = np.empty(ar.shape, dtype=np.intp)
                inv_idx[perm] = iflag
                ret += (inv_idx,)
            if return_counts:
                idx = np.concatenate(np.nonzero(flag) + ([ar.size],))
                ret += (np.diff(idx),)
        return ret

    def colorEncode(self, labelmap, colors, mode='RGB'):
        labelmap = labelmap.astype('int')
        labelmap_rgb = np.zeros((labelmap.shape[0], labelmap.shape[1], 3),
                                dtype=np.uint8)
        for label in self.unique(labelmap):
            if label < 0:
                continue
            labelmap_rgb += (labelmap == label)[:, :, np.newaxis] * \
                            np.tile(colors[label],
                                    (labelmap.shape[0], labelmap.shape[1], 1))

        if mode == 'BGR':
            return labelmap_rgb[:, :, ::-1]
        else:
            return labelmap_rgb

    def show_result(self, img, pred, save_path=None):
        pred = np.int32(pred)
        # colorize prediction
        pred_color = self.colorEncode(pred, self.colors)
        pil_img = img.convert('RGBA')
        pred_color = Image.fromarray(pred_color).convert('RGBA')
        im_vis = Image.blend(pil_img, pred_color, 0.6)
        if save_path is not None:
            im_vis.save(save_path)
            # Image.fromarray(im_vis).save(save_path)
        else:
            plt.imshow(im_vis)


def get_dims_with_exclusion(dim, exclude=None):
    dims = list(range(dim))
    if exclude is not None:
        dims.remove(exclude)

    return dims


def save_checkpoint(net, checkpoints_path, epoch=None, prefix='', verbose=True, multi_gpu=False):
    if epoch is None:
        checkpoint_name = 'last_checkpoint.pth'
    else:
        checkpoint_name = f'{epoch:03d}.pth'

    if prefix:
        checkpoint_name = f'{prefix}_{checkpoint_name}'

    if not checkpoints_path.exists():
        checkpoints_path.mkdir(parents=True)

    checkpoint_path = checkpoints_path / checkpoint_name
    if verbose:
        logger.info(f'Save checkpoint to {str(checkpoint_path)}')

    net = net.module if multi_gpu else net
    torch.save({'state_dict': net.state_dict(),
                'config': net._config
                }, str(checkpoint_path))


def get_bbox_from_mask(mask):
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return rmin, rmax, cmin, cmax


def expand_bbox(bbox, expand_ratio, min_crop_size=None):
    rmin, rmax, cmin, cmax = bbox
    rcenter = 0.5 * (rmin + rmax)
    ccenter = 0.5 * (cmin + cmax)
    height = expand_ratio * (rmax - rmin + 1)
    width = expand_ratio * (cmax - cmin + 1)
    if min_crop_size is not None:
        height = max(height, min_crop_size)
        width = max(width, min_crop_size)

    rmin = int(round(rcenter - 0.5 * height))
    rmax = int(round(rcenter + 0.5 * height))
    cmin = int(round(ccenter - 0.5 * width))
    cmax = int(round(ccenter + 0.5 * width))

    return rmin, rmax, cmin, cmax


def clamp_bbox(bbox, rmin, rmax, cmin, cmax):
    return (max(rmin, bbox[0]), min(rmax, bbox[1]),
            max(cmin, bbox[2]), min(cmax, bbox[3]))


def get_bbox_iou(b1, b2):
    h_iou = get_segments_iou(b1[:2], b2[:2])
    w_iou = get_segments_iou(b1[2:4], b2[2:4])
    return h_iou * w_iou


def get_segments_iou(s1, s2):
    a, b = s1
    c, d = s2
    intersection = max(0, min(b, d) - max(a, c) + 1)
    union = max(1e-6, max(b, d) - min(a, c) + 1)
    return intersection / union


def get_labels_with_sizes(x):
    obj_sizes = np.bincount(x.flatten())
    labels = np.nonzero(obj_sizes)[0].tolist()
    labels = [x for x in labels if x != 0]
    return labels, obj_sizes[labels].tolist()
