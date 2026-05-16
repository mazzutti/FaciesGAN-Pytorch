from collections.abc import Callable
from typing import Any, cast

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

    def update_weights(
        self,
        loss_matrix: torch.Tensor,
        shared_layer: nn.Module,
        forward_fn: Callable[[dict[str, torch.Tensor]], torch.Tensor] | None = None,
        scale_idx: int | None = None,
        accumulate_only: bool = False,
    ) -> torch.Tensor:
        """Ajusta os pesos w_i baseado no conflito de gradientes entre escalas e tipos.

        Se `forward_fn` for fornecido, utiliza vectorização (torch.func) para calcular
        gradientes de todas as tasks em paralelo. Caso contrário, cai no modo sequencial.
        Se `accumulate_only` for True, calcula os gradientes mas não executa o step do otimizador.
        Se `scale_idx` for fornecido, calcula o GradNorm apenas para aquela escala.
        """
        self.optimizer.zero_grad()

        # 1. Preparação das losses individuais (flattened)
        # Garantimos que estamos em float32 para estabilidade e para evitar conflitos com AMP/BFloat16
        loss_matrix = loss_matrix.float()

        if scale_idx is not None:
            # No modo incremental, pegamos apenas a linha desta escala.
            curr_l_tensor = loss_matrix[scale_idx]
            w_flat = self.w[scale_idx]
        else:
            # Modo padrão: rastreia toda a pirâmide (memória intensiva)
            curr_l_tensor = loss_matrix.view(-1)
            w_flat = self.w.view(-1)

        # 2. Inicialização do L0 (primeira época ou nova escala)
        # Note: No modo progressivo, novas escalas podem aparecer depois.
        # Inicializamos apenas os valores que ainda são zero.
        with torch.no_grad():
            if scale_idx is not None:
                num_tasks = curr_l_tensor.size(0)
                start = scale_idx * num_tasks
                end = (scale_idx + 1) * num_tasks
                # Inicializa apenas se o L0 desta escala ainda for zero
                if self.l0[start:end].abs().sum() < 1e-6:
                    self.l0[start:end].copy_(curr_l_tensor.detach().clamp(min=1e-4))
            elif not self.l0_initialized:
                # Fallback para modo global
                self.l0.copy_(curr_l_tensor.detach().clamp(min=1e-4))
            
            self.l0_initialized.fill_(True)

        # 3. Cálculo das normas dos gradientes (G_i)
        # Pegamos o L0 correspondente
        if scale_idx is not None:
            num_tasks = curr_l_tensor.size(0)
            l0_scale = self.l0[scale_idx * num_tasks : (scale_idx + 1) * num_tasks].to(curr_l_tensor.device)
        else:
            l0_scale = self.l0.to(curr_l_tensor.device)
        if forward_fn is not None:
            # --- MODO VECTORIZADO (torch.func) ---
            from torch.func import vjp, vmap

            # Extraímos apenas os parâmetros que querem gradientes (shared_layer)
            # Para evitar que o vjp tente derivar de tudo no modelo.
            # Usamos uma tupla ordenada de parâmetros para maior robustez com torch.func
            p_named = list(shared_layer.named_parameters())
            p_names = [n for n, p in p_named if p.requires_grad]
            p_tensors = tuple(p for n, p in p_named if p.requires_grad)

            if not p_tensors:
                return torch.tensor(0.0, device=loss_matrix.device, dtype=torch.float32)

            # Definimos uma função que mapeia parâmetros para a matriz de losses
            def compute_loss_vector(params_tuple: tuple[torch.Tensor, ...]) -> torch.Tensor:
                # Reconstrói o dicionário esperado pelo forward_fn
                params_dict = {name: p for name, p in zip(p_names, params_tuple)}
                return forward_fn(params_dict).reshape(-1)

            # Usamos vjp para obter a função que calcula o produto Jacobiano-Vetor
            _loss_vec, vjp_fn = cast(
                tuple[torch.Tensor, Any], vjp(compute_loss_vector, p_tensors)
            )

            # Para obter as normas individuais |dL_i / dTheta|, precisamos passar
            # vetores "one-hot" (e_i) para o vjp_fn.
            eye = torch.eye(
                len(curr_l_tensor),
                device=loss_matrix.device,
                dtype=_loss_vec.dtype,
            )

            # Para evitar erros de vmap ou de segunda derivada em kernels específicos
            # (como aten::cudnn_grid_sampler_backward), e considerando que temos
            # apenas 8 tasks por escala, podemos calcular a norma de cada uma de forma
            # puramente sequencial. Isso é extremamente rápido, consome menos memória
            # e evita qualquer incompatibilidade de vmap ou de batching no vjp.
            norms_list: list[torch.Tensor] = []
            for v in eye:
                vjp_res = vjp_fn(v)
                if not vjp_res or vjp_res[0] is None:
                    norms_list.append(torch.tensor(0.0, device=v.device, dtype=torch.float32))
                    continue

                grads = vjp_res[0]
                # Detaching the gradients prevents any second-order autograd (double backward)
                # through the generator's parameters, avoiding compatibility errors with
                # operators that don't support it (like grid_sample).
                sq_norms = [g.detach().pow(2).sum() for g in grads if g is not None]
                
                if not sq_norms:
                    norms_list.append(torch.tensor(0.0, device=v.device, dtype=torch.float32))
                else:
                    norms_list.append(torch.sqrt(torch.stack(sq_norms).sum()).float())

            norms = torch.stack(norms_list)
        else:
            # --- MODO SEQUENCIAL (Fallback) ---
            params = list(shared_layer.parameters())
            norms_tensor = torch.zeros_like(curr_l_tensor)

            for i in range(len(curr_l_tensor)):
                grad = torch.autograd.grad(
                    curr_l_tensor[i],
                    params,
                    retain_graph=True,
                    create_graph=False,
                )
                norm_sq = torch.stack([g.pow(2).sum() for g in grad]).sum()
                norms_tensor[i] = torch.sqrt(norm_sq)
            norms = norms_tensor

        # 4. Cálculo do Target baseado na taxa de aprendizado relativa (ri)
        with torch.no_grad():
            r_i: torch.Tensor = curr_l_tensor / l0_scale
            r_i = r_i / r_i.mean().clamp(min=1e-6)  # Normalização relativa

            mean_norm: torch.Tensor = norms.mean()
            constant_term: torch.Tensor = mean_norm * (r_i**self.alpha)

        # 5. Loss do GradNorm (L_grad)
        # Note: 'norms' aqui já inclui o peso 'w' se calculado via sequential,
        # ou é a norma pura se calculada via vjp (dependendo de como forward_fn é implementado).
        # Para consistência, o vjp acima calcula dL_i/dTheta.
        # GradNorm original define Gi = w_i(t) * |grad_Theta(w_i(t) * L_i(t))|
        # Mas na prática, w_i(t) é tratado como constante para o gradiente de w.
        loss_grad = torch.nn.functional.l1_loss(norms * w_flat, constant_term)

        # Escala a loss se estivermos acumulando de várias escalas
        if accumulate_only:
            loss_grad = loss_grad / len(self.scales)

        loss_grad.backward()  # type: ignore

        if not accumulate_only:
            self.optimizer.step()  # pyright: ignore[reportUnknownMemberType]

            # 6. Renormalização dos pesos (Soma = N_tasks total)
            with torch.no_grad():
                # Renormalizamos todos os pesos para manter a escala global da learning rate
                n_tasks_total = self.w.numel()
                self.w.data *= n_tasks_total / self.w.sum()

        return loss_grad
