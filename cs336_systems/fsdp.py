from __future__ import annotations

import types
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn

from cs336_basics.model import Embedding, Linear


@dataclass
class _ShardedWeightInfo: 
    """Metadata for one unique sharded weight. padded 用于特殊情况补齐参数 Parameter fp32存储"""

    parameter: nn.Parameter
    full_shape: torch.Size
    full_numel: int
    shard_numel: int
    padded_numel: int


class _FSDPLinearFunction(torch.autograd.Function):
    """
    Linear operation whose master weight is stored as a flat FP32 shard.

    Forward:
      all-gather shards -> reconstruct full weight -> compute.

    Backward:
      all-gather shards -> compute grad_input;
      compute the local full-weight gradient -> reduce-scatter it.
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        local_weight: torch.Tensor,
        fsdp: "FSDP",
        info: _ShardedWeightInfo,
        module_index: int,
    ) -> torch.Tensor:
        full_weight = fsdp._acquire_full_weight(info, module_index)

        ctx.fsdp = fsdp
        ctx.info = info
        ctx.save_for_backward(x)

        output = torch.einsum("...i,oi->...o", x, full_weight)

        # Once this layer has finished, prefetch the layer two positions ahead.
        fsdp._after_forward_module(module_index)
        return output

    @staticmethod
    def backward(
        ctx,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, None, None, None]:
        (x,) = ctx.saved_tensors
        fsdp: FSDP = ctx.fsdp
        info: _ShardedWeightInfo = ctx.info

        grad_input = None
        if ctx.needs_input_grad[0]:
            # Linear backward needs the full weight to compute grad_input.
            full_weight = fsdp._gather_weight_sync(info)
            grad_input = torch.einsum(
                "...o,oi->...i",
                grad_output,
                full_weight,
            )

        local_weight_grad = None
        if ctx.needs_input_grad[1]:
            x_2d = x.reshape(-1, x.shape[-1])
            grad_output_2d = grad_output.reshape(
                -1,
                grad_output.shape[-1],
            )

            # Keep this computation in the compute dtype, matching the
            # mixed-precision reference behavior, then communicate/store the
            # resulting master gradient in FP32.
            full_weight_grad = grad_output_2d.transpose(0, 1).matmul(x_2d)
            local_weight_grad = fsdp._reduce_scatter_gradient(
                info,
                full_weight_grad,
            )

        return grad_input, local_weight_grad, None, None, None


class _FSDPEmbeddingFunction(torch.autograd.Function):
    """Embedding operation backed by a flat FP32 weight shard."""

    @staticmethod
    def forward(
        ctx,
        token_ids: torch.Tensor,
        local_weight: torch.Tensor,
        fsdp: "FSDP",
        info: _ShardedWeightInfo,
        module_index: int,
    ) -> torch.Tensor:
        full_weight = fsdp._acquire_full_weight(info, module_index)

        ctx.fsdp = fsdp
        ctx.info = info
        ctx.save_for_backward(token_ids)

        output = full_weight[token_ids, :]

        fsdp._after_forward_module(module_index)
        return output

    @staticmethod
    def backward(
        ctx,
        grad_output: torch.Tensor,
    ) -> tuple[None, torch.Tensor | None, None, None, None]:
        (token_ids,) = ctx.saved_tensors
        fsdp: FSDP = ctx.fsdp
        info: _ShardedWeightInfo = ctx.info

        local_weight_grad = None
        if ctx.needs_input_grad[1]:
            # Weight values are not required for an embedding backward pass.
            full_weight_grad = torch.zeros(
                info.full_shape,
                device=grad_output.device,
                dtype=grad_output.dtype,
            )
            full_weight_grad.index_add_(
                0,
                token_ids.reshape(-1),
                grad_output.reshape(-1, grad_output.shape[-1]),
            )
            local_weight_grad = fsdp._reduce_scatter_gradient(
                info,
                full_weight_grad,
            )

        return None, local_weight_grad, None, None, None


class FSDP(nn.Module):
    """
    A small educational Fully-Sharded Data Parallel implementation.

    Linear and Embedding weights:
      * are stored permanently as flat FP32 shards;
      * are all-gathered only when needed for forward/backward;
      * receive reduce-scattered FP32 gradients.

    Other parameters, such as RMSNorm weights, remain replicated and have
    their gradients all-reduced in finish_gradient_synchronization().
    """

    def __init__(self, module: nn.Module, compute_dtype: torch.dtype | None = None,):
        super().__init__()

        self.module = module
        self.compute_dtype = compute_dtype

        if dist.is_available() and dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            # This fallback is convenient for single-process debugging.
            self.rank = 0
            self.world_size = 1

        # module id -> (weight metadata, forward-order index, layer kind)
        # 记录每个sub module的信息
        self._module_infos: dict[
            int,
            tuple[_ShardedWeightInfo, int, str],
        ] = {}

        # Original Parameter id -> metadata. This preserves tied weights. 
        # 为了处理共享权重 用id()方法识别参数类 防止重复分片
        self._infos_by_original_param_id: dict[
            int,
            _ShardedWeightInfo,
        ] = {}

        # Linear/Embedding modules in registration/expected execution order. 这个顺序主要是后面预取的时候用
        self._ordered_sharded_modules: list[nn.Module] = []

        # Non-FSDP parameters, such as normalization weights.
        self._replicated_parameters: list[nn.Parameter] = []

        # module index -> (Work, gathered buffer, send buffer) 保存已经发起、但可能尚未完成的异步权重通信。
        # Keeping send_buffer alive is important for asynchronous collectives.
        self._pending_prefetches: dict[
            int,
            tuple[Any, torch.Tensor, torch.Tensor],
        ] = {}
        # 其实只分片 emd linear 其他层全跳过
        sharded_modules = [
            submodule
            for submodule in self.module.modules()
            if isinstance(submodule, (Linear, Embedding))
        ]

        for module_index, submodule in enumerate(sharded_modules):
            original_weight = submodule.weight
            original_parameter_id = id(original_weight)
            # 检测是不是和其他模块共享权重的
            info = self._infos_by_original_param_id.get(
                original_parameter_id
            )
            if info is None:
                info = self._make_weight_shard(original_weight)
                self._infos_by_original_param_id[
                    original_parameter_id
                ] = info

            # If two modules shared the original Parameter, they now share the
            # same local shard Parameter as well.
            submodule.weight = info.parameter

            kind = (
                "linear"
                if isinstance(submodule, Linear)
                else "embedding"
            )
            self._module_infos[id(submodule)] = (
                info,
                module_index,
                kind,
            )
            self._ordered_sharded_modules.append(submodule)

            self._replace_forward(submodule, kind)
        # 收集所有info id
        sharded_parameter_ids = {
            id(info.parameter)
            for info in self._infos_by_original_param_id.values()
        }

        seen_parameter_ids: set[int] = set()
        for parameter in self.module.parameters():
            parameter_id = id(parameter)
            if parameter_id in seen_parameter_ids:
                continue
            seen_parameter_ids.add(parameter_id)

            if parameter_id not in sharded_parameter_ids:
                self._replicated_parameters.append(parameter)

    def _make_weight_shard(
        self,
        original_weight: nn.Parameter,
    ) -> _ShardedWeightInfo:
        """
        Flatten, zero-pad, and shard one weight across all ranks.

        The permanent master shard is always FP32.
        """
        flat_weight = (
            original_weight.detach()
            .reshape(-1)
            .to(dtype=torch.float32)
        )

        full_numel = flat_weight.numel()
        shard_numel = (
            full_numel + self.world_size - 1
        ) // self.world_size
        padded_numel = shard_numel * self.world_size

        if padded_numel == full_numel:
            padded_weight = flat_weight
        else:
            padded_weight = torch.zeros(
                padded_numel,
                device=flat_weight.device,
                dtype=torch.float32,
            )
            padded_weight[:full_numel].copy_(flat_weight)

        shard_start = self.rank * shard_numel
        local_shard = padded_weight[
            shard_start : shard_start + shard_numel
        ].clone()

        shard_parameter = nn.Parameter(
            local_shard,
            requires_grad=original_weight.requires_grad,
        )

        return _ShardedWeightInfo(
            parameter=shard_parameter,
            full_shape=original_weight.shape,
            full_numel=full_numel,
            shard_numel=shard_numel,
            padded_numel=padded_numel,
        )

    def _replace_forward(
        self,
        submodule: nn.Module,
        kind: str,
    ) -> None:
        """
        Keep the original module object/type, but route its operation through
        a custom autograd function that understands the sharded weight.
        """
        if kind == "linear":

            def linear_forward(
                module: Linear,
                x: torch.Tensor,
                _fsdp: FSDP = self,
            ) -> torch.Tensor:
                info, module_index, _ = _fsdp._module_infos[id(module)]
                return _FSDPLinearFunction.apply(
                    x,
                    module.weight,
                    _fsdp,
                    info,
                    module_index,
                )

            submodule.forward = types.MethodType(
                linear_forward,
                submodule,
            )
            return

        def embedding_forward(
            module: Embedding,
            token_ids: torch.Tensor,
            _fsdp: FSDP = self,
        ) -> torch.Tensor:
            info, module_index, _ = _fsdp._module_infos[id(module)]
            return _FSDPEmbeddingFunction.apply(
                token_ids,
                module.weight,
                _fsdp,
                info,
                module_index,
            )

        submodule.forward = types.MethodType(
            embedding_forward,
            submodule,
        )

    def _communication_shard(
        self,
        info: _ShardedWeightInfo,
    ) -> torch.Tensor:
        """
        Return the tensor to communicate.

        Master shards stay FP32. When compute_dtype is supplied, casting is
        performed before all-gather so communication bandwidth is reduced.
        """
        shard = info.parameter.detach()

        if self.compute_dtype is not None:
            shard = shard.to(self.compute_dtype)

        return shard.contiguous()

    def _start_weight_prefetch(
        self,
        module_index: int,
    ) -> None:
        if self.world_size == 1:
            return

        if not (
            0 <= module_index < len(self._ordered_sharded_modules)
        ):
            return

        if module_index in self._pending_prefetches:
            return

        submodule = self._ordered_sharded_modules[module_index]
        info, _, _ = self._module_infos[id(submodule)]

        send_buffer = self._communication_shard(info)
        gathered_buffer = torch.empty(
            info.padded_numel,
            device=send_buffer.device,
            dtype=send_buffer.dtype,
        )

        work = dist.all_gather_into_tensor(
            gathered_buffer,
            send_buffer,
            async_op=True,
        )

        self._pending_prefetches[module_index] = (
            work,
            gathered_buffer,
            send_buffer,
        )

    def _acquire_full_weight(
        self,
        info: _ShardedWeightInfo,
        module_index: int,
    ) -> torch.Tensor:
        """
        Wait for a prefetched all-gather, or gather synchronously if the layer
        was not prefetched.
        """
        pending = self._pending_prefetches.pop(
            module_index,
            None,
        )

        if pending is None:
            return self._gather_weight_sync(info)

        work, gathered_buffer, _send_buffer = pending
        work.wait()

        return gathered_buffer[
            : info.full_numel
        ].view(info.full_shape)

    def _gather_weight_sync(
        self,
        info: _ShardedWeightInfo,
        *,
        use_compute_dtype: bool = True,
    ) -> torch.Tensor:
        """
        同步收集完整权重。

        use_compute_dtype=True:
            用于前向和反向计算。
            如果指定了 compute_dtype，则先将分片转成低精度再通信。

        use_compute_dtype=False:
            用于 checkpoint 或参数正确性检查。
            直接通信 FP32 主权重分片，不损失精度。
        """
        if use_compute_dtype:
            send_buffer = self._communication_shard(info)
        else:
            send_buffer = info.parameter.detach().contiguous()

        if self.world_size == 1:
            return send_buffer[
                : info.full_numel
            ].view(info.full_shape)

        gathered_buffer = torch.empty(
            info.padded_numel,
            device=send_buffer.device,
            dtype=send_buffer.dtype,
        )

        dist.all_gather_into_tensor(
            gathered_buffer,
            send_buffer,
        )

        return gathered_buffer[
            : info.full_numel
        ].view(info.full_shape)
    def _after_forward_module(
        self,
        completed_module_index: int,
    ) -> None:
        # Gather layer i + 2 only after layer i has completed.
        self._start_weight_prefetch(completed_module_index + 2)

    def _reduce_scatter_gradient(
        self,
        info: _ShardedWeightInfo,
        full_gradient: torch.Tensor,
    ) -> torch.Tensor:
        """
        Sum full local gradients across ranks and leave each rank with only the
        gradient corresponding to its FP32 master shard.
        """
        flat_gradient = (
            full_gradient.reshape(-1)
            .to(dtype=torch.float32)
            .contiguous()
        )

        if info.padded_numel == info.full_numel:
            padded_gradient = flat_gradient
        else:
            padded_gradient = torch.zeros(
                info.padded_numel,
                device=flat_gradient.device,
                dtype=torch.float32,
            )
            padded_gradient[: info.full_numel].copy_(flat_gradient)

        if self.world_size == 1:
            return padded_gradient[: info.shard_numel].clone()

        local_gradient = torch.empty(
            info.shard_numel,
            device=flat_gradient.device,
            dtype=torch.float32,
        )

        dist.reduce_scatter_tensor(
            local_gradient,
            padded_gradient,
            op=dist.ReduceOp.SUM,
        )

        # Match DDP's gradient averaging semantics.
        local_gradient.div_(self.world_size)
        return local_gradient

    def _finish_stale_prefetches(self) -> None:
        """
        A dynamic/branched model might not consume every prefetched layer.
        Finish those collectives before beginning another forward pass.
        """
        for work, _gathered, _send in self._pending_prefetches.values():
            work.wait()

        self._pending_prefetches.clear()

    def forward(self, *inputs, **kwargs):
        self._finish_stale_prefetches()

        # The first two sharded layers cannot be prefetched by earlier layers.
        self._start_weight_prefetch(0)
        self._start_weight_prefetch(1)

        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self) -> None:
        """
        Synchronize gradients of parameters that are intentionally replicated.

        Sharded Linear/Embedding gradients have already been reduce-scattered
        by their custom backward functions.
        """
        if self.world_size == 1:
            return

        for parameter in self._replicated_parameters:
            if not parameter.requires_grad or parameter.grad is None:
                continue

            dist.all_reduce(
                parameter.grad,
                op=dist.ReduceOp.SUM,
            )
            parameter.grad.div_(self.world_size)

    @torch.no_grad()
    def gather_full_params(self) -> dict[str, torch.Tensor]:
        """
        Reconstruct a full state dictionary for testing/checkpoint inspection.

        Replicated parameters are returned directly; sharded weights are
        all-gathered and reshaped to their original forms.
        """
        self._finish_stale_prefetches()

        result: dict[str, torch.Tensor] = {}
        named_modules = dict(self.module.named_modules())

        for name, parameter in self.module.named_parameters():
            module_path, separator, parameter_name = name.rpartition(".")
            owner_module = (
                named_modules[module_path]
                if separator
                else self.module
            )

            module_info = self._module_infos.get(id(owner_module))
            if (
                module_info is not None
                and parameter_name == "weight"
            ):
                info, _, _ = module_info
                result[name] = (
                    self._gather_weight_sync(info, use_compute_dtype=False).clone()
                )
            else:
                result[name] = parameter.detach().clone()

        return result