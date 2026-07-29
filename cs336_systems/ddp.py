import torch
import torch.distributed as dist
import torch.nn as nn
import time

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

        self.last_sync_time = time.perf_counter() - start