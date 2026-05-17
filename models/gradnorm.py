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
            # Cria um dicionário mapeando cada chave para seu valor na matriz
            keys = MetricKey.generator_keys()
            d: dict[MetricKey, torch.Tensor] = {
                key: loss_matrix[i, j] for j, key in enumerate(keys)
            }
            # Calcula o total para esta escala (soma da linha i)
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

    def update_weights_from_graph(
        self,
        curr_losses: torch.Tensor,
        shared_layer: nn.Module,
        scale_idx: int,
    ) -> torch.Tensor:
        """Ajusta os pesos w usando o grafo de autograd já existente (sem segundo forward pass).

        Calcula as normas de gradiente de cada task diretamente sobre ``curr_losses``
        (que ainda possui grafo de autograd ativo) usando ``torch.autograd.grad`` com
        ``retain_graph=True``.  Isso elimina o custo de um segundo forward pass
        completo pelo gerador (sem ``functional_call`` / VJP).

        Deve ser chamado **antes** de ``total_loss.backward()`` enquanto o grafo
        de autograd do forward pass do generator ainda estiver ativo.

        Parameters
        ----------
        curr_losses : torch.Tensor
            Vetor 1D de shape ``(num_tasks,)`` com as losses individuais, em float32,
            **com grafo de autograd ativo** (i.e., sem `.detach()`).
        shared_layer : nn.Module
            Camada proxy cujos parâmetros serão usados para calcular as normas.
        scale_idx : int
            Índice da escala atual (0-based) para indexar ``self.w`` e ``self.l0``.

        Returns
        -------
        torch.Tensor
            Escalar com a loss do GradNorm usada para atualizar os pesos.
        """
        self.optimizer.zero_grad()

        # Garantir float32 para estabilidade numérica
        curr_l_tensor = curr_losses.float()
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
        # Usa o grafo existente de autograd (sem segundo forward pass).
        # retain_graph=True é necessário pois iremos chamar backward() de várias tasks.
        # A última chamada pode usar retain_graph=False (mas usamos True para segurança).
        params = list(shared_layer.parameters())
        if not params:
            return torch.tensor(0.0, device=curr_losses.device, dtype=torch.float32)

        norms_list: list[torch.Tensor] = []
        n = len(curr_l_tensor)
        for i in range(n):
            try:
                grads = torch.autograd.grad(
                    curr_l_tensor[i],
                    params,
                    retain_graph=True,   # grafo será reutilizado para as próximas tasks e para total_loss.backward()
                    create_graph=False,  # não precisa de segunda derivada
                    allow_unused=True,
                )
                sq_norms = [g.detach().pow(2).sum() for g in grads if g is not None]
                if sq_norms:
                    norms_list.append(torch.sqrt(torch.stack(sq_norms).sum()).float())
                else:
                    norms_list.append(torch.tensor(0.0, device=curr_losses.device, dtype=torch.float32))
            except Exception:
                norms_list.append(torch.tensor(0.0, device=curr_losses.device, dtype=torch.float32))

        norms = torch.stack(norms_list)

        # Cálculo do Target baseado na taxa de aprendizado relativa (ri)
        with torch.no_grad():
            r_i: torch.Tensor = curr_l_tensor.detach() / l0_scale
            r_i = r_i / r_i.mean().clamp(min=1e-6)

            mean_norm: torch.Tensor = norms.mean()
            constant_term: torch.Tensor = mean_norm * (r_i**self.alpha)

        # Loss do GradNorm (L_grad) — diferencia apenas w_flat
        loss_grad = torch.nn.functional.l1_loss(norms * w_flat, constant_term)

        loss_grad.backward()  # type: ignore

        self.optimizer.step()  # pyright: ignore[reportUnknownMemberType]

        # Renormalização dos pesos (Soma = N_tasks total)
        with torch.no_grad():
            n_tasks_total = self.w.numel()
            self.w.data *= n_tasks_total / self.w.sum()

        return loss_grad
