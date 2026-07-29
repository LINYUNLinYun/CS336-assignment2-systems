import torch
import torch.distributed as dist
import torch.nn as nn
import time
from torch._utils import _flatten_dense_tensors
from torch._utils import _unflatten_dense_tensors

class NaiveDDP(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        self.last_sync_time = 0.0

        if dist.is_initialized():
            self._broadcast_parameters()

    def _broadcast_parameters(self):
        for p in self.module.parameters():
            dist.broadcast(p.data, src=0)



    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def synchronize_gradients_flat(self):
        grads = []

        params = []
        """
        All-reduce every parameter gradient.
        """
        if not dist.is_initialized():
            return

        world_size = dist.get_world_size()

        for p in self.module.parameters():

            if p.grad is None:
                continue
            
            grads.append(p.grad)
            params.append(p)
        flat_grad = _flatten_dense_tensors(grads)

        dist.all_reduce(flat_grad, op=dist.ReduceOp.SUM, async_op=False)

        flat_grad/=(world_size)

        synced = _unflatten_dense_tensors(flat_grad,grads,)

        for p,g in zip(params, synced):
            p.grad.copy_(g)

    def synchronize_gradients(self):
        """
        All-reduce every parameter gradient.
        """
        if not dist.is_initialized():
            return

        world_size = dist.get_world_size()
        start = time.perf_counter()

        for p in self.module.parameters():

            if p.grad is None:
                continue

            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM, async_op=False)

            p.grad.div_(world_size)
