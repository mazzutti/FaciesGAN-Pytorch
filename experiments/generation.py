"""Sample generation logic for experiments."""

import logging
from pathlib import Path

import numpy as np
import torch

from config import DomainConfig
from device import device_manager
from enums import ChannelKey, DataFiles, SplitKey
from models.facies_gan import FaciesGAN
from models.utils import split_facies_rp
from options import TrainingOptions

logger = logging.getLogger(__name__)


def _generate_samples(
    gpu_id: int | None,
    model_path: str,
    opts: TrainingOptions,
    how_many: int,
    wells_pyramid: tuple[torch.Tensor, ...],
    seismic_pyramid: tuple[torch.Tensor, ...],
    channels: dict[ChannelKey, int],
    gen_output: str,
    start_index: int,
    seen_indices: list[int] | None = None,
    eps: float = DomainConfig.EPSILON,
) -> tuple[
    list[np.ndarray],
    list[np.ndarray],
    list[np.ndarray],
    list[np.ndarray],
    list[np.ndarray],
    torch.Tensor,
]:
    """Generate facies (and rock physics when enabled) on a single device.

    Returns
    -------
    tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray], torch.Tensor]
        ``(facies_arrays, ip_arrays, is_arrays, vpvs_arrays, seismic_arrays, mask_indexes)``.
        *ip_arrays*, *is_arrays*, *vpvs_arrays* and *seismic_arrays* are empty when rock physics is not enabled.
    """

    device = device_manager.get_or_initialize(gpu_id)

    # Disable torch.compile for generation/inference to avoid the heavy compilation
    # startup overhead and prevent concurrent thread-compilation crashes in ThreadPoolExecutor.
    import copy

    opts_eval = copy.deepcopy(opts)
    opts_eval.compile_backend = False

    model = FaciesGAN(options=opts_eval, channels=channels)
    model.load(model_path, load_discriminator=False, load_wells=False)

    has_rock_physics = opts.use_rock_physics
    facies_ch = model.num_facies_channels

    max_scale = len(model.noise_amps) - 1
    print(
        f"    [{device}] Generating {how_many} samples for {model_path} (max_scale={max_scale})"
    )

    # Check spatial dimensions from first scale noise spec to be sure
    spatial_shape = model.get_noise_shape(max_scale, use_base_channel=False)
    print(f"    [{device}] Target spatial shape: {spatial_shape}")

    all_facies: list[np.ndarray] = []
    all_ip: list[np.ndarray] = []
    all_is: list[np.ndarray] = []
    all_vpvs: list[np.ndarray] = []
    all_seismic: list[np.ndarray] = []
    all_mi: list[torch.Tensor] = []

    # Batch size for generation to avoid OOM
    batch_size = 20

    gen_output_path = Path(gen_output)
    facies_dir = gen_output_path / DataFiles.FACIES.name.lower()
    ip_dir = gen_output_path / DataFiles.Ip.name.lower()
    is_dir = gen_output_path / DataFiles.Is.name.lower()
    vpvs_dir = gen_output_path / DataFiles.VP_VS.name.lower()
    seismic_dir = gen_output_path / DataFiles.SEISMIC.name.lower()
    facies_dir.mkdir(parents=True, exist_ok=True)
    if has_rock_physics:
        ip_dir.mkdir(parents=True, exist_ok=True)
        is_dir.mkdir(parents=True, exist_ok=True)
        vpvs_dir.mkdir(parents=True, exist_ok=True)
        seismic_dir.mkdir(parents=True, exist_ok=True)

    # These are constant for the entire generation run — build once outside the loop
    if seen_indices:
        well_choices = seen_indices
        print(
            f"    [{device}] Using {len(well_choices)} seen indices provided to generator."
        )
    else:
        well_choices = (
            opts.wells_mask_columns
            if hasattr(opts, "wells_mask_columns") and opts.wells_mask_columns
            else (
                list(range(wells_pyramid[max_scale].shape[0])) if wells_pyramid else [0]
            )
        )
    wells_dict = (
        {i: wells_pyramid[i] for i in range(len(wells_pyramid))}
        if (wells_pyramid and opts.use_wells)
        else {}
    )
    seismic_dict = (
        {i: seismic_pyramid[i] for i in range(len(seismic_pyramid))}
        if (seismic_pyramid and opts.use_seismic)
        else {}
    )

    noise_amps = [torch.tensor(a, device=device) for a in model.noise_amps]

    for off in range(0, how_many, batch_size):
        count = min(batch_size, how_many - off)

        # Ensure we use deterministic indexing even when batching.
        # Cycle through well_choices (which may be seen_indices).
        start_in_choices = (start_index + off) % len(well_choices)
        mi_idx = [
            well_choices[(start_in_choices + i) % len(well_choices)]
            for i in range(count)
        ]
        mi = torch.tensor(mi_idx, dtype=torch.long, device=device)
        all_mi.append(mi.cpu())

        with torch.no_grad():
            noises = model.get_pyramid_noise(
                max_scale, mi, wells_dict, seismic_dict, rec=opts.rec
            )
            for j, g in enumerate(model.generator(noises, noise_amps)):
                # generator may be typed imprecisely; ensure we treat outputs as tensors
                idx = start_index + off + j + 1

                # Use unified channel splitting logic
                split = split_facies_rp(
                    g.unsqueeze(0), num_facies=facies_ch, has_rp=has_rock_physics
                )
                facies_t = split[SplitKey.FACIES]
                rp_t = split[SplitKey.ROCK_PHYSICS]

                # Keep one-hot probabilities for embedding analysis (high-fidelity)
                # facies_t may be None in some configurations; skip if so
                if facies_t is None:
                    continue
                facies_np = device_manager.to_numpy(facies_t.squeeze(0))
                all_facies.append(facies_np)

                # Save categorical indices to disk (more compact for users)
                # Move argmax result to CPU non-blocking and convert to numpy
                facies_idx_t = torch.argmax(facies_t.squeeze(0), dim=0)
                facies_idx = (
                    device_manager.to_cpu(facies_idx_t, non_blocking=True)
                    .numpy()
                    .astype(np.int64)
                )
                np.save(
                    facies_dir / f"{DataFiles.FACIES.name.lower()}_{idx:04d}.npy",
                    facies_idx,
                )

                if has_rock_physics and rp_t is not None:
                    # rp_t has channels for active properties only.
                    # For disk saving, denormalize to actual physical units using centralized PhysicsState.
                    phys_dict = model.physics_state.denormalize_rock_physics(rp_t)

                    active_properties: list[str] = []
                    if getattr(opts, "use_ip", True):
                        active_properties.append("Ip")
                    if getattr(opts, "use_is", True):
                        active_properties.append("Is")
                    if getattr(opts, "use_vpvs", True):
                        active_properties.append("VP_VS")

                    for ch_idx, prop_name in enumerate(active_properties):
                        prop_norm = rp_t[:, ch_idx : ch_idx + 1, ...]
                        # For manifold learning and embedding comparisons, keep features in normalized range [-1, 1]
                        if prop_name == "Ip":
                            all_ip.append(device_manager.to_numpy(prop_norm.squeeze(0)))
                            ip_phys = device_manager.to_numpy(
                                phys_dict[DataFiles.Ip.name].squeeze(0)
                            )
                            np.save(
                                ip_dir / f"{DataFiles.Ip.name.lower()}_{idx:04d}.npy",
                                ip_phys,
                            )
                        elif prop_name == "Is":
                            all_is.append(device_manager.to_numpy(prop_norm.squeeze(0)))
                            is_phys = device_manager.to_numpy(
                                phys_dict[DataFiles.Is.name].squeeze(0)
                            )
                            np.save(
                                is_dir / f"{DataFiles.Is.name.lower()}_{idx:04d}.npy",
                                is_phys,
                            )
                        elif prop_name == "VP_VS":
                            all_vpvs.append(
                                device_manager.to_numpy(prop_norm.squeeze(0))
                            )
                            vpvs_phys = device_manager.to_numpy(
                                phys_dict[DataFiles.VP_VS.name].squeeze(0)
                            )
                            np.save(
                                vpvs_dir
                                / f"{DataFiles.VP_VS.name.lower()}_{idx:04d}.npy",
                                vpvs_phys,
                            )

                    # Also synthetic seismic (requires Ip)
                    if getattr(opts, "use_ip", True):
                        seismic: torch.Tensor = model.get_synthetic_seismic(
                            g.unsqueeze(0)
                        )
                        all_seismic.append(device_manager.to_numpy(seismic.squeeze(0)))
                        np.save(
                            seismic_dir
                            / f"{DataFiles.SEISMIC.name.lower()}_{idx:04d}.npy",
                            device_manager.to_numpy(seismic.squeeze(0)),
                        )

    return (
        all_facies,
        all_ip,
        all_is,
        all_vpvs,
        all_seismic,
        torch.cat(all_mi),
    )


def generate_variant(
    model_path: str,
    opts: TrainingOptions,
    how_many: int,
    wells_pyramid: tuple[torch.Tensor, ...],
    seismic_pyramid: tuple[torch.Tensor, ...],
    channels: dict[ChannelKey, int],
    gen_output: str,
    seen_indices: list[int] | None = None,
) -> tuple[
    list[np.ndarray],
    list[np.ndarray],
    list[np.ndarray],
    list[np.ndarray],
    list[np.ndarray],
    torch.Tensor,
]:
    """Load a trained model and generate facies (and rock physics) samples.

    Returns
    -------
    tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray], torch.Tensor]
        ``(facies, ip, is_data, vpvs, seismic, mask_indexes)``.
    """
    num_gpus = device_manager.accelerator_count
    if num_gpus >= 2:
        from concurrent.futures import Future, ThreadPoolExecutor

        # Pre-initialize CUDA contexts on every GPU from the main thread
        # so worker threads don't hit "operation not permitted" errors.
        device_manager.warmup_all_accelerators()

        chunk_per_gpu = how_many // num_gpus
        remainder = how_many % num_gpus
        futures: list[
            Future[
                tuple[
                    list[np.ndarray],
                    list[np.ndarray],
                    list[np.ndarray],
                    list[np.ndarray],
                    list[np.ndarray],
                    torch.Tensor,
                ]
            ]
        ] = []
        with ThreadPoolExecutor(max_workers=num_gpus) as pool:
            start = 0
            for gpu_id in range(num_gpus):
                count = chunk_per_gpu + (1 if gpu_id < remainder else 0)
                if count == 0:
                    continue
                futures.append(
                    pool.submit(
                        _generate_samples,
                        gpu_id,
                        model_path,
                        opts,
                        count,
                        wells_pyramid,
                        seismic_pyramid,
                        channels,
                        gen_output,
                        start,
                        seen_indices=seen_indices,
                    )
                )
                start += count
        facies: list[np.ndarray] = []
        ip: list[np.ndarray] = []
        is_data: list[np.ndarray] = []
        vpvs: list[np.ndarray] = []
        seismic: list[np.ndarray] = []
        mask_indexes_list: list[torch.Tensor] = []
        for fut in futures:
            (
                variant_facies,
                variant_ip,
                variant_is,
                variant_vpvs,
                variant_seismic,
                variant_mi,
            ) = fut.result()
            facies.extend(variant_facies)
            ip.extend(variant_ip)
            is_data.extend(variant_is)
            vpvs.extend(variant_vpvs)
            seismic.extend(variant_seismic)
            mask_indexes_list.append(variant_mi)

        return facies, ip, is_data, vpvs, seismic, torch.cat(mask_indexes_list)
    else:
        return _generate_samples(
            None,
            model_path,
            opts,
            how_many,
            wells_pyramid,
            seismic_pyramid,
            channels,
            gen_output,
            0,
            seen_indices=seen_indices,
        )
