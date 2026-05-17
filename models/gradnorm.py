from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
from training.metrics import GeneratorMetrics
from enums import MetricKey
from device import device_manager


class GradNorm(nn.Module):
    scales: list[int]
    alpha: float
    w: nn.Parameter
    optimizer: optim.Adam
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
        self.optimizer: optim.Adam = optim.Adam([self.w], lr=lr)

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
            )

        return metrics_dict

    def update_weights_from_isolated_forward(
        self,
        scale_block: nn.Module,
        z_in: torch.Tensor,
        task_losses_fn: Any,
        scale_idx: int,
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

        params = list(scale_block.parameters())
        if not params:
            return torch.tensor(0.0, device=z_in.device, dtype=torch.float32)

        # Forward pass mínimo: apenas pelo bloco atual, com z_in já detachado.
        # O grafo de autograd é pequeno: z_in → scale_block → fake_local.
        fake_local = scale_block(z_in)

        # Computar as losses sobre fake_local (grafo pequeno, só 1 bloco)
        curr_l_tensor = task_losses_fn(fake_local).float()
        w_flat = self.w[scale_idx]

        # Inicialização do L0
        with torch.no_grad():
            num_tasks = curr_l_tensor.size(0)
            start = scale_idx * num_tasks
            end = (scale_idx + 1) * num_tasks
            if self.l0[start:end].abs().sum() < 1e-6:
                self.l0[start:end].copy_(curr_l_tensor.detach().clamp(min=1e-4))
            self.l0_initialized.fill_(True)

        l0_scale = self.l0[start:end].to(curr_l_tensor.device)

        # --- Calcular normas dos gradientes de cada task ---
        # O grafo é pequeno (apenas 1 bloco), então retain_graph=True é barato.
        norms_list: list[torch.Tensor] = []
        n = len(curr_l_tensor)
        for i in range(n):
            try:
                grads = torch.autograd.grad(
                    curr_l_tensor[i],
                    params,
                    retain_graph=(i < n - 1),  # libera o grafo na última task
                    create_graph=False,
                    allow_unused=True,
                )
                sq_norms = [g.detach().pow(2).sum() for g in grads if g is not None]
                if sq_norms:
                    norms_list.append(torch.sqrt(torch.stack(sq_norms).sum()).float())
                else:
                    norms_list.append(torch.tensor(0.0, device=z_in.device, dtype=torch.float32))
            except Exception:
                norms_list.append(torch.tensor(0.0, device=z_in.device, dtype=torch.float32))

        norms = torch.stack(norms_list)

        # Cálculo do Target baseado na taxa de aprendizado relativa (ri)
        with torch.no_grad():
            r_i: torch.Tensor = curr_l_tensor.detach() / l0_scale
            r_i = r_i / r_i.mean().clamp(min=1e-6)

            mean_norm: torch.Tensor = norms.mean()
            constant_term: torch.Tensor = mean_norm * (r_i**self.alpha)

        # Loss do GradNorm — diferencia apenas w_flat (não os parâmetros do bloco)
        loss_grad = torch.nn.functional.l1_loss(norms * w_flat, constant_term)

        loss_grad.backward()  # type: ignore

        self.optimizer.step()  # pyright: ignore[reportUnknownMemberType]

        # Renormalização dos pesos (Soma = N_tasks total)
        with torch.no_grad():
            n_tasks_total = self.w.numel()
            self.w.data *= n_tasks_total / self.w.sum()

        return loss_grad
