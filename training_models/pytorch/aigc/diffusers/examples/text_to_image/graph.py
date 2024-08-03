#
# Copyright 2023 Enflame. All Rights Reserved.
# Here is a wrap of torch_gcu.core.experiment.py/model.py
#
import os
import torch
import torch_gcu
from torch_gcu.core.device import _device_t
from torch_gcu._GCUC import JitRunMode
from typing import Any

"""
Note [JitRunMode]
SYNC, run graph synchronously,
ASYNC, run graph asynchronously, deliver graph without waiting, lead higher HBM peak
SAFEASYNC, run graph asynchronously, wait last graph finish before deliver graph
"""

def split_graph_by_tensors(tensors, mode=JitRunMode.SAFEASYNC):
    """Create a graph using given tensors as outputs

    Args:
      tensors (List/tuple[torch.Tensor]): List or tuple of `torch.Tensor`s to materialize.
        For each Tensor `t` in the list, `t.device` must be a `GCU` device.
        Can not empty.
      mode(JitRunMode): Run graph mode. See Note [JitRunMode].
    """
    torch_gcu.unlazy(tensors, mode)

def split_graph_auto(device: _device_t = None, mode=JitRunMode.SAFEASYNC):
    """Create and run a graph using all lived tensors on device as outputs

    Args:
      device: Specific device, if :attr:`device` is ``None``, this will use the current default GCU device
    device
      mode(JitRunMode): Run graph mode. See Note [JitRunMode].
    """
    if device is None:
        device = torch_gcu.current_device()
    torch_gcu.sync_lived_tensor(device, mode)

class SplitGraphNow(torch_gcu.core.experiment.AutoSyncGraph):
    r"""split a graph, auto or manually.

    Args:
        threshold (int): the threshold of how many torch op to accumulate before run the graph.
            <  0: do nothing, return
            == 0: auto split a graph
            >  0: split a graph with op number
        device: Specific device, if :attr:`device` is ``None``, this will use the current default GCU device
        mode(JitRunMode): Run graph mode. See Note [JitRunMode].
    """
    def __init__(self, threshold: int = -1, device: _device_t = None, mode=JitRunMode.SAFEASYNC) -> None:
        if  threshold <= 0:
            self.threshold = threshold
            self.device = device
            self.mode = mode
        else:
            super().__init__(threshold, device, mode)

    def __enter__(self) -> None:
        if self.threshold < 0:
            return
        elif self.threshold == 0:
            split_graph_auto(self.device, self.mode)
        else:
            super().__enter__()

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        if self.threshold <= 0:
            return
        super().__exit__(exc_type, exc_value, traceback)
