import torch
import os
import sys
import logging
import logging.handlers
import json
import psutil
import subprocess
from functools import wraps
import importlib.util
from torch._six import inf
import torch
if torch.cuda.is_available():
    import torch.distributed as dist
else:
    import torch_gcu.distributed as dist

def clip_grad_norm_(parameters, max_norm, norm_type: float = 2.0) -> torch.Tensor:
    r"""Clips gradient norm of an iterable of parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.

    Arguments:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.

    Returns:
        Total norm of the parameters (viewed as a single vector).
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    device = parameters[0].grad.device
    max_norm = torch.tensor(float(max_norm)).to(device)
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)

    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)

    clip_coef = max_norm / (total_norm + torch.tensor(1e-6).to(device))
    clip_coef = torch.lt(clip_coef,1.0).float()*clip_coef + torch.ge(clip_coef,1.0).float()
    for p in parameters:
        p.grad.detach().mul_(clip_coef.to(p.grad.device))
    return total_norm
