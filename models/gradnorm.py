import logging
from collections.abc import Mapping
from typing import Any, cast

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from training.metrics import GeneratorMetrics
from enums import MetricKey
from device import device_manager

logger = logging.getLogger(__name__)


class GradNormAdam(optim.Adam):
    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        all_params: list[torch.Tensor] = []
        for group in self.param_groups:
            group_params = cast(list[torch.Tensor], group["params"])
            all_params.extend(group_params)

        if "state" in state_dict:
            for k, state in list(state_dict["state"].items()):
                try:
                    param_idx = int(k)
                except ValueError:
                    continue

                if 0 <= param_idx < len(all_params):
                    param = all_params[param_idx]
                    for key in ["exp_avg", "exp_avg_sq"]:
                        if key in state:
                            old_t = state[key]
                            if old_t.shape != param.shape:
                                new_t = torch.zeros(
                                    param.shape, dtype=old_t.dtype, device=old_t.device
                                )
                                if old_t.dim() == 2 and param.dim() == 2:
                                    old_num_scales, old_num_keys = old_t.shape
                                    new_num_scales, new_num_keys = param.shape
                                    common_scales = min(old_num_scales, new_num_scales)
                                    common_keys = min(old_num_keys, new_num_keys)
                                    new_t[:common_scales, :common_keys] = old_t[
                                        :common_scales, :common_keys
                                    ]
                                elif old_t.dim() == 1 and param.dim() == 1:
                                    common_len = min(len(old_t), len(param))
                                    new_t[:common_len] = old_t[:common_len]
                                state[key] = new_t
        super().load_state_dict(state_dict)


class GradNorm(nn.Module):
    scales: list[int]
    alpha: float
    w: nn.Parameter
    optimizer: GradNormAdam
    l0: torch.Tensor
    l0_initialized: torch.Tensor

    def __init__(
        self,
        scales: list[int],
        alpha: float = 0.15,
        lr: float = 0.0005,
        chunk_size: int | None = None,
    ):
        super().__init__()
        self.scales = scales
        self.alpha = alpha
        self.chunk_size = chunk_size

        # Geramos uma matriz de pesos (Scales x Loss_Keys)
        # Inicializados em 1.0 para começar com o balanceamento original
        self.w = nn.Parameter(
            torch.ones(
                len(scales),
                len(MetricKey.generator_keys()),
                device=device_manager.device,
            )
        )

        # Otimizador dedicado para os pesos das losses
        self.optimizer = GradNormAdam([self.w], lr=lr)

        # Buffer for initial losses (L0) for normalization
        self.register_buffer(
            "l0",
            torch.zeros(
                len(scales) * len(MetricKey.generator_keys()),
                dtype=torch.float32,
                device=device_manager.device,
            ),
        )
        self.register_buffer(
            "l0_initialized",
            torch.tensor(False, dtype=torch.bool, device=device_manager.device),
        )

    def load_state_dict(
        self,
        state_dict: Mapping[str, Any],
        strict: bool = True,
        assign: bool = False,
    ) -> Any:
        # Adapt 'w' and 'l0' shapes if they mismatch from checkpoint
        adapted_state_dict = dict(state_dict)
        old_w = adapted_state_dict.get("w")
        old_l0 = adapted_state_dict.get("l0")

        if old_w is not None and old_w.shape != self.w.shape:
            new_w = torch.ones_like(self.w)
            if old_w.dim() == 2 and self.w.dim() == 2:
                old_num_scales, old_num_keys = old_w.shape
                new_num_scales, new_num_keys = self.w.shape
                common_scales = min(old_num_scales, new_num_scales)
                common_keys = min(old_num_keys, new_num_keys)
                new_w[:common_scales, :common_keys] = old_w[
                    :common_scales, :common_keys
                ]
                adapted_state_dict["w"] = new_w

                if old_l0 is not None:
                    try:
                        old_l0_reshaped = old_l0.view(old_num_scales, old_num_keys)
                        new_l0_reshaped = torch.zeros(
                            (new_num_scales, new_num_keys),
                            dtype=self.l0.dtype,
                            device=self.l0.device,
                        )
                        new_l0_reshaped[:common_scales, :common_keys] = old_l0_reshaped[
                            :common_scales, :common_keys
                        ]
                        adapted_state_dict["l0"] = new_l0_reshaped.view(-1)
                    except Exception:
                        adapted_state_dict["l0"] = torch.zeros_like(self.l0)
            else:
                adapted_state_dict["w"] = self.w.clone()

        elif old_l0 is not None and old_l0.shape != self.l0.shape:
            adapted_state_dict["l0"] = torch.zeros_like(self.l0)

        return super().load_state_dict(
            adapted_state_dict,
            strict=strict,
            assign=assign,
        )

    def get_weighted_loss(self, loss_matrix: torch.Tensor) -> torch.Tensor:
        """Calcula a soma ponderada de todas as losses de todas as escalas: sum(W * L)."""
        return (self.w * loss_matrix).sum()

    def to_metrics_dict(self, loss_matrix: torch.Tensor) -> dict[int, GeneratorMetrics]:
        """Converte a matriz de losses de volta para um dicionário de GeneratorMetrics."""
        metrics_dict: dict[int, GeneratorMetrics] = {}

        for i, s in enumerate(self.scales):
            keys = MetricKey.generator_keys()
            d: dict[MetricKey, torch.Tensor] = {
                key: loss_matrix[i, j] for j, key in enumerate(keys)
            }
            d[MetricKey.G_TOTAL] = loss_matrix[i].sum()
            metrics_dict[s] = GeneratorMetrics(
                total=d[MetricKey.G_TOTAL],
                fake=d[MetricKey.G_FAKE],
                rec_facies=d[MetricKey.G_REC_FACIES],
                well=d[MetricKey.G_WELL],
                div=d[MetricKey.G_DIV],
                rec_rock_physics=d[MetricKey.G_REC_ROCK_PHYSICS],
                tv=d[MetricKey.G_TV],
                elastic=d[MetricKey.G_ELASTIC],
                seismic=d[MetricKey.G_SEISMIC],
                integrated_rpm=d[MetricKey.G_INTEGRATED_RPM],
            )

        return metrics_dict

    def update_weights_from_isolated_forward(
        self,
        scale_block: nn.Module,
        z_in: torch.Tensor,
        task_losses_fn: Any,
        scale_idx: int,
        adv_penalty: float = 10.0,
    ) -> torch.Tensor:
        """Ajusta os pesos w usando um forward pass mínimo e isolado pelo bloco do scale atual.

        Estratégia eficiente: em vez de percorrer o grafo completo (discriminador +
        pirâmide inteira), faz um forward pass mínimo somente pelo bloco do scale
        atual, usando ``z_in`` já detachado de todos os scales anteriores.
        Isso limita o backward a apenas 1 bloco convolucional, não à pirâmide toda.

        Parameters
        ----------
        scale_block : nn.Module
            Bloco de geração do scale atual (já desembrulhado de torch.compile/DDP).
        z_in : torch.Tensor
            Entrada do bloco do scale atual, já detachada dos scales anteriores
            (``requires_grad=True`` para permitir backward).
        task_losses_fn : callable
            Função ``(fake_out: Tensor) -> Tensor`` que recebe a saída do gerador
            e retorna o vetor 1D de losses ``(num_tasks,)`` em float32.
        scale_idx : int
            Índice da escala atual para indexar ``self.w`` e ``self.l0``.

        Returns
        -------
        torch.Tensor
            Escalar com a loss do GradNorm.
        """
        self.optimizer.zero_grad()

        params = [p for p in scale_block.parameters() if p.requires_grad]
        if not params:
            return torch.tensor(0.0, device=z_in.device, dtype=torch.float32)

        # Check if task_losses_fn supports task_idx parameter (cached)
        if not hasattr(self, "_has_task_idx"):
            has_task_idx = False
            _func = task_losses_fn
            if not hasattr(_func, "__code__") and hasattr(_func, "__call__"):
                _func = _func.__call__
            if hasattr(_func, "__code__"):
                _code = _func.__code__
                has_task_idx = "task_idx" in _code.co_varnames[: _code.co_argcount]
            self._has_task_idx = has_task_idx
        else:
            has_task_idx = self._has_task_idx

        # Forward pass mínimo: apenas pelo bloco atual, com z_in já detachado.
        # Para evitar conflito com donated buffers do torch.compile/functorch,
        # não reutilizamos o mesmo grafo para múltiplas chamadas de grad.
        with torch.no_grad():
            fake_local = scale_block(z_in)
            if has_task_idx:
                curr_l_tensor = task_losses_fn(fake_local, task_idx=None).float()
            else:
                curr_l_tensor = task_losses_fn(fake_local).float()

        # Sincronizar as losses entre as instâncias DDP para balanceamento global
        if device_manager.is_distributed:
            dist.all_reduce(curr_l_tensor, op=dist.ReduceOp.AVG)  # type: ignore

        w_flat = self.w[scale_idx]

        # Inicialização do L0
        num_tasks = curr_l_tensor.size(0)
        start = scale_idx * num_tasks
        end = (scale_idx + 1) * num_tasks

        # Transform raw losses for L0 and r_i calculation to ensure all are positive and decreasing.
        # WGAN adversarial loss (index 0) can be negative and unbounded; we map it to exp(loss / penalty).
        with torch.no_grad():
            curr_l_transformed = curr_l_tensor.clone()
            curr_l_transformed[0] = torch.exp(curr_l_tensor[0] / adv_penalty)

            if self.l0[start:end].abs().sum() < 1e-6:
                self.l0[start:end].copy_(curr_l_transformed.detach().clamp(min=1e-4))
            self.l0_initialized.fill_(True)

        l0_scale = self.l0[start:end].to(curr_l_tensor.device)

        # --- Calcular normas dos gradientes de cada task ---
        # Cada task usa um forward isolado para manter retain_graph=False sempre.
        # Norms are computed on the original losses curr_l_i because we want the gradients of actual training losses.
        norms_list: list[torch.Tensor] = []
        n = len(curr_l_tensor)
        curr_l_cpu = curr_l_tensor.cpu()
        for i in range(n):
            try:
                if curr_l_cpu[i] == 0.0:
                    norms_list.append(
                        torch.tensor(0.0, device=z_in.device, dtype=torch.float32)
                    )
                    continue

                fake_i = scale_block(z_in)
                if has_task_idx:
                    curr_l_i = task_losses_fn(fake_i, task_idx=i).float()
                else:
                    curr_l_i = task_losses_fn(fake_i).float()
                grads = cast(
                    tuple[torch.Tensor | None, ...],
                    torch.autograd.grad(
                        curr_l_i[i],
                        params,
                        retain_graph=False,
                        create_graph=False,
                        allow_unused=True,
                    ),
                )
                sq_norms = [g.detach().pow(2).sum() for g in grads if g is not None]
                if sq_norms:
                    norms_list.append(torch.sqrt(torch.stack(sq_norms).sum()).float())
                else:
                    norms_list.append(
                        torch.tensor(0.0, device=z_in.device, dtype=torch.float32)
                    )
            except Exception:
                norms_list.append(
                    torch.tensor(0.0, device=z_in.device, dtype=torch.float32)
                )

        norms = torch.stack(norms_list)

        # Sincronizar as normas dos gradientes entre as instâncias DDP
        if device_manager.is_distributed:
            dist.all_reduce(norms, op=dist.ReduceOp.AVG)  # type: ignore

        # Cálculo do Target baseado na taxa de aprendizado relativa (ri)
        with torch.no_grad():
            r_i: torch.Tensor = curr_l_transformed.detach() / l0_scale
            r_i = r_i / r_i.mean().clamp(min=1e-6)

            mean_norm: torch.Tensor = norms.mean()
            # Use a.abs() to prevent NaN gradients with negative losses (like WGAN adversarial loss)
            constant_term: torch.Tensor = mean_norm * (r_i.abs() ** self.alpha)

        # Loss do GradNorm — diferencia apenas w_flat (não os parâmetros do bloco)
        loss_grad = torch.nn.functional.l1_loss(norms * w_flat, constant_term)

        try:
            loss_grad.backward()  # type: ignore
            self.optimizer.step()  # pyright: ignore[reportUnknownMemberType]
        except Exception as e:
            # Fallback em caso de erro no autograd do GradNorm (ex: grafo quebrado)
            logger.warning(
                f"GradNorm weight update failed (scale {scale_idx}): {e}", exc_info=True
            )

        # Renormalização dos pesos (Soma = N_tasks total)
        with torch.no_grad():
            n_tasks_total = self.w.numel()
            self.w.data *= n_tasks_total / self.w.sum().clamp(min=1e-6)

        return loss_grad
