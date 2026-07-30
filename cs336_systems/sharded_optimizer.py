from __future__ import annotations

from collections import defaultdict
from typing import Any, Type

import torch
import torch.distributed as dist
from torch.optim import Optimizer


class ShardedOptimizer(Optimizer):
    """
    Optimizer-state-sharding wrapper.

    每个参数只会被分配给一个 rank：
    - 该 rank 的本地 optimizer 负责更新这个参数，并保存其 optimizer state；
    - optimizer.step() 完成后，该 rank 将更新后的参数广播给其他 rank；
    - 所有 rank 最终仍持有完整且一致的模型参数。

    注意：
    这个类只负责 optimizer state sharding 和参数同步。
    在调用 step() 之前，各 rank 的梯度应当已经通过 DDP 等方式完成同步。
    """

    def __init__( self, params, optimizer_cls: Type[Optimizer], **kwargs: Any,
    ):
        # 支持非分布式环境，方便单进程调试。
        if dist.is_available() and dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1

        self.optimizer_cls = optimizer_cls

        # id(parameter) -> 负责更新该参数的 rank。
        self._param_owners: dict[int, int] = {}

        # 用于以 round-robin 的方式分配新参数。 轮流分配
        self._next_param_index = 0

        # 将 params 统一转换为参数组形式。
        full_param_groups = self._normalize_param_groups(params)

        # 为当前 rank 构造其负责的参数分片。
        local_param_groups = [
            self._make_local_param_group(param_group)
            for param_group in full_param_groups
        ]

        # 真正执行参数更新的本地 optimizer。
        # 它只会看到当前 rank 所拥有的参数。
        self._local_optimizer = optimizer_cls(
            local_param_groups,
            **kwargs,
        )

        # Optimizer.__init__ 会调用 self.add_param_group()。
        # 使用该标记避免在初始化父类时重复向本地 optimizer 添加参数组。
        self._in_super_init = True

        super().__init__(
            full_param_groups,
            defaults=dict(self._local_optimizer.defaults),
        )

        self._in_super_init = False

        # 让外层 optimizer 暴露的 state 就是本地 optimizer 的分片状态。
        self.state = self._local_optimizer.state

    @staticmethod
    def _normalize_param_groups(params) -> list[dict[str, Any]]:
        """
        将以下两种输入统一转换成参数组：

        1. model.parameters()
        2. [
               {"params": module1.parameters(), "lr": ...},
               {"params": module2.parameters(), "lr": ...},
           ]
        """
        if isinstance(params, torch.Tensor):
            raise TypeError(
                "params argument must be an iterable of Tensors "
                "or an iterable of parameter-group dictionaries"
            )

        params = list(params)

        if len(params) == 0:
            raise ValueError("optimizer got an empty parameter list")

        # 参数组形式。
        if isinstance(params[0], dict):
            if not all(isinstance(item, dict) for item in params):
                raise TypeError(
                    "params must contain either only Tensors "
                    "or only parameter-group dictionaries"
                )

            param_groups: list[dict[str, Any]] = []

            for item in params:
                group = dict(item)

                if "params" not in group:
                    raise KeyError("parameter group must contain a 'params' entry")

                group_params = group["params"]

                if isinstance(group_params, torch.Tensor):
                    group["params"] = [group_params]
                else:
                    group["params"] = list(group_params)

                param_groups.append(group)

            return param_groups

        # 普通参数 iterable 形式。
        if any(isinstance(item, dict) for item in params):
            raise TypeError(
                "params must contain either only Tensors "
                "or only parameter-group dictionaries"
            )

        return [{"params": params}]

    def _make_local_param_group(
        self,
        param_group: dict[str, Any],
    ) -> dict[str, Any]:
        """
        根据参数所有权，为当前 rank 创建一个参数组分片。

        参数按照出现顺序使用 round-robin 分配：

            parameter_index % world_size

        同一个 Parameter 对象只会分配一次，这也能正确处理 tied weights。
        """
        local_group = {
            key: value
            for key, value in param_group.items()
            if key != "params"
        }

        local_params = []

        for param in param_group["params"]:
            if not isinstance(param, torch.Tensor):
                raise TypeError(
                    "optimizer can only optimize Tensors, "
                    f"but one of the params is {type(param)}"
                )

            param_id = id(param)

            owner = self._param_owners.get(param_id)

            # 第一次遇到这个参数时，为它分配 owner。
            if owner is None:
                owner = self._next_param_index % self.world_size

                self._param_owners[param_id] = owner
                self._next_param_index += 1

            if owner == self.rank:
                local_params.append(param)

        local_group["params"] = local_params
        return local_group

    def add_param_group(
        self,
        param_group: dict[str, Any],
    ) -> None:
        """
        添加新的参数组。

        这个方法会在两种情况下调用：

        1. Optimizer 父类初始化；
        2. 训练过程中用户主动调用 optimizer.add_param_group()。
        """
        # 父类 Optimizer.__init__ 正在注册完整参数组。
        # 此时本地 optimizer 已经在 __init__ 中构造完毕，
        # 不应再次向它添加相同参数组。
        if getattr(self, "_in_super_init", False):
            super().add_param_group(param_group)
            return

        group = dict(param_group)

        if "params" not in group:
            raise KeyError("parameter group must contain a 'params' entry")

        group_params = group["params"]

        if isinstance(group_params, torch.Tensor):
            group["params"] = [group_params]
        else:
            group["params"] = list(group_params)

        # 外层 optimizer 保存完整参数组。
        # 这样继承的 zero_grad() 会清除所有模型参数的梯度，
        # 而不仅是当前 rank 拥有的参数。
        super().add_param_group(group)

        # 为当前 rank 创建新参数组的本地分片。
        # 使用 self.param_groups[-1]，其中已经包含父类补齐的默认配置。
        local_group = self._make_local_param_group(
            self.param_groups[-1]
        )

        self._local_optimizer.add_param_group(local_group)

    def _sync_group_options_to_local_optimizer(self) -> None:
        """
        将外层参数组的超参数同步给本地 optimizer。

        这样用户修改 optimizer.param_groups，或者使用学习率 scheduler 时，
        本地 optimizer 也会使用最新的学习率等配置。
        """
        for full_group, local_group in zip(
            self.param_groups,
            self._local_optimizer.param_groups,
            strict=True,
        ):
            for key, value in full_group.items():
                if key != "params":
                    local_group[key] = value

    def step(
        self,
        closure=None,
        **kwargs: Any,
    ):
        """
        更新当前 rank 拥有的参数，然后同步所有模型参数。
        """
        self._sync_group_options_to_local_optimizer()

        # 当前 rank 只更新自己的参数分片。
        loss = self._local_optimizer.step(
            closure=closure,
            **kwargs,
        )

        if self.world_size > 1:
            # tied weights 或重复参数只广播一次。
            seen: set[int] = set()

            with torch.no_grad():
                # 所有 rank 必须按照完全相同的顺序调用 collective。
                for param_group in self.param_groups:
                    for param in param_group["params"]:
                        param_id = id(param)

                        if param_id in seen:
                            continue

                        seen.add(param_id)

                        owner = self._param_owners[param_id]

                        # owner 拥有更新后的参数值，其余 rank 接收该值。
                        # 这个函数既可以发送也可以接收
                        dist.broadcast(
                            param.data,
                            src=owner,
                        )

        return loss

    def load_state_dict(
        self,
        state_dict: dict[str, Any],
    ) -> None:
        """
        加载当前 rank 的分片 optimizer state。

        父类加载状态后，需要重新让本地 optimizer 引用对应的本地状态。
        """
        super().load_state_dict(state_dict)

        local_state = defaultdict(dict)

        for param_group in self._local_optimizer.param_groups:
            for param in param_group["params"]:
                if param in self.state:
                    local_state[param] = self.state[param]

        self._local_optimizer.state = local_state
        self.state = self._local_optimizer.state

        self._sync_group_options_to_local_optimizer()