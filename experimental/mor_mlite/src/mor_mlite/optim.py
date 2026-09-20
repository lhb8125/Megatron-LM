"""Reference FP32-master AdamW matching the ZeRO-1 numerical contract."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch


class MasterWeightAdamW:
    """AdamW over FP32 master weights with explicit BF16 model synchronization.

    This is the single-rank oracle, not a distributed optimizer.  MLite's
    distributed optimizer owns/shards the same logical state on multi-rank
    runs and performs reduce-scatter/all-gather at the step boundary.
    """

    def __init__(
        self,
        parameters: Iterable[torch.nn.Parameter],
        *,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        self.model_parameters = [parameter for parameter in parameters if parameter.requires_grad]
        self.master_parameters = [
            torch.nn.Parameter(parameter.detach().float().clone(), requires_grad=True)
            for parameter in self.model_parameters
        ]
        self.optimizer = torch.optim.AdamW(
            self.master_parameters,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            foreach=False,
            fused=False,
        )

    def zero_grad(self, *, set_to_none: bool = True) -> None:
        for parameter in self.model_parameters:
            parameter.grad = None if set_to_none else torch.zeros_like(parameter)
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def step(self, *, clip_grad: float | None = None) -> torch.Tensor:
        for model_parameter, master_parameter in zip(
            self.model_parameters, self.master_parameters, strict=True
        ):
            master_parameter.grad = (
                None
                if model_parameter.grad is None
                else model_parameter.grad.detach().float().clone()
            )
        grads = [p for p in self.master_parameters if p.grad is not None]
        if grads:
            if clip_grad is None:
                grad_norm = torch.linalg.vector_norm(
                    torch.stack([torch.linalg.vector_norm(p.grad) for p in grads])
                )
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(grads, clip_grad)
        else:
            grad_norm = torch.tensor(0.0)
        self.optimizer.step()
        with torch.no_grad():
            for model_parameter, master_parameter in zip(
                self.model_parameters, self.master_parameters, strict=True
            ):
                model_parameter.copy_(master_parameter.to(model_parameter.dtype))
        return torch.as_tensor(grad_norm).detach()

    def state_dict(self) -> dict[str, Any]:
        return {
            "optimizer": self.optimizer.state_dict(),
            "master_weights": [parameter.detach().cpu() for parameter in self.master_parameters],
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.optimizer.load_state_dict(state["optimizer"])
        masters = state["master_weights"]
        if len(masters) != len(self.master_parameters):
            raise ValueError("master weight count differs from current model")
        with torch.no_grad():
            for model_parameter, master_parameter, value in zip(
                self.model_parameters, self.master_parameters, masters, strict=True
            ):
                master_parameter.copy_(value.to(master_parameter))
                model_parameter.copy_(master_parameter.to(model_parameter.dtype))


__all__ = ["MasterWeightAdamW"]
