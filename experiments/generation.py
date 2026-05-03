"""Sample generation logic for experiments."""

import os
from typing import cast

import numpy as np
import torch

from datasets.data_files import DataFiles
from models.facies_gan import TorchFaciesGAN
from models.utils import calculate_synthetic_seismic, split_facies_rp
from options import TrainingOptions


def _generate_on_device(
    device: torch.device,
    model_path: str,
    opts: TrainingOptions,
    how_many: int,
    wells_pyramid: tuple[torch.Tensor, ...],
    seismic_pyramid: tuple[torch.Tensor, ...],
    noise_channels: int,
    gen_output: str,
    start_index: int,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], torch.Tensor]:
    """Generate facies (and rock physics when enabled) on a single device.

    Returns
    -------
    tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], torch.Tensor]
        ``(facies_arrays, ip_arrays, seismic_arrays, mask_indexes)``.
        *ip_arrays* and *seismic_arrays* are empty when rock physics is not enabled.
    """
    import random as _rng

    model = TorchFaciesGAN(options=opts, device=device, noise_channels=noise_channels)
    model.load(model_path, load_discriminator=False, load_wells=False)

    has_rock_physics = getattr(opts, "use_rock_physics", False)
    facies_ch = model.num_facies_classes

    max_scale = len(model.noise_amps) - 1
    print(
        f"    [{device}] Generating {how_many} samples for {model_path} (max_scale={max_scale})"
    )

    # Check spatial dimensions from first scale noise spec to be sure
    spatial_shape = model.get_noise_shape(max_scale, use_base_channel=False)
    print(f"    [{device}] Target spatial shape: {spatial_shape}")

    all_facies: list[np.ndarray] = []
    all_ip: list[np.ndarray] = []
    all_seismic: list[np.ndarray] = []
    all_mi: list[torch.Tensor] = []

    # Batch size for generation to avoid OOM
    batch_size = 20

    facies_dir = os.path.join(gen_output, DataFiles.FACIES.name.lower())
    ip_dir = os.path.join(gen_output, DataFiles.Ip.name.lower())
    seismic_dir = os.path.join(gen_output, DataFiles.SEISMIC.name.lower())
    os.makedirs(facies_dir, exist_ok=True)
    if has_rock_physics:
        os.makedirs(ip_dir, exist_ok=True)
        os.makedirs(seismic_dir, exist_ok=True)

    for off in range(0, how_many, batch_size):
        chunk = min(batch_size, how_many - off)
        # Use specific wells if provided, otherwise pick randomly from available conditioning
        well_choices = (
            opts.wells_mask_columns
            if hasattr(opts, "wells_mask_columns") and opts.wells_mask_columns
            else (
                list(range(wells_pyramid[max_scale].shape[0])) if wells_pyramid else [0]
            )
        )
        mi = torch.tensor(
            [_rng.choice(well_choices) for _ in range(chunk)], device=device
        )
        wells_dict = (
            {i: wells_pyramid[i] for i in range(len(wells_pyramid))}
            if wells_pyramid
            else {}
        )
        seismic_dict = (
            {i: seismic_pyramid[i] for i in range(len(seismic_pyramid))}
            if seismic_pyramid
            else {}
        )
        noises = model.get_pyramid_noise(
            max_scale, mi, wells_dict, seismic_dict, rec=opts.rec
        )
        with torch.no_grad():
            for j, g in enumerate(
                model.generator(noises, model.get_noise_amplitude(max_scale))
            ):
                # generator may be typed imprecisely; ensure we treat outputs as tensors
                g = cast(torch.Tensor, g)
                idx = start_index + off + j + 1

                # Use unified channel splitting logic
                split = split_facies_rp(
                    g.unsqueeze(0), num_facies=facies_ch, has_rp=has_rock_physics
                )
                facies_t = split["facies"]
                rp_t = split["rock_physics"]

                # Keep one-hot probabilities for embedding analysis (high-fidelity)
                # facies_t may be None in some configurations; skip if so
                if facies_t is None:
                    continue
                facies_np = facies_t.squeeze(0).cpu().numpy()
                all_facies.append(facies_np)

                # Save categorical indices to disk (more compact for users)
                facies_idx = (
                    torch.argmax(facies_t.squeeze(0), dim=0)
                    .cpu()
                    .numpy()
                    .astype(np.int64)
                )
                np.save(
                    os.path.join(facies_dir, f"generated_{DataFiles.FACIES.name.lower()}_{idx}.npy"), facies_idx
                )

                if has_rock_physics and rp_t is not None:
                    # Ip is the first channel of RP
                    ip_norm = rp_t[:, 0:1, ...]
                    # Map to [0, 1] for saving as IP array (Legacy behavior)
                    ip_arr = (ip_norm.squeeze(0).squeeze(0).cpu().numpy() + 1.0) / 2.0
                    ip_arr = np.clip(ip_arr, 0.0, 1.0).astype(np.float32)

                    # Unified Seismic Modeling (Matches Trainer)
                    # Use model buffers for physics parameters
                    rho_mean = cast(torch.Tensor, model.rho_mean)
                    vp_min = cast(torch.Tensor, model.vp_min)
                    vp_max = cast(torch.Tensor, model.vp_max)
                    ip_max = cast(torch.Tensor, model.ip_max)
                    ip_min = cast(torch.Tensor, model.ip_min)
                    seis_min = cast(torch.Tensor, model.seis_min)
                    seis_max = cast(torch.Tensor, model.seis_max)

                    # Dynamic Vp estimation (Matches Trainer)
                    ip_phys: torch.Tensor = ((ip_norm + 1) / 2) * (
                        ip_max - ip_min
                    ) + ip_min
                    vp_phys = ip_phys / rho_mean
                    vp_mean = torch.mean(vp_phys).clamp(vp_min, vp_max)

                    # Ensure wavelet dt is a tensor to match calculate_synthetic_seismic signature
                    wavelet_dt_t = torch.tensor(
                        opts.wavelet_dt,
                        device=ip_norm.device,
                        dtype=ip_norm.dtype,
                    )
                    synth = calculate_synthetic_seismic(
                        ip_norm,
                        model.wavelet_t,
                        wavelet_dt_t,
                        cast(torch.Tensor, model.dz_pyramid)[max_scale],
                        vp_mean,
                        vp_min,
                        ip_min,
                        ip_max,
                        seis_min,
                        seis_max,
                    )
                    syn_seismic = (
                        synth.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)
                    )

                    all_ip.append(ip_arr)
                    all_seismic.append(syn_seismic)

                    np.save(os.path.join(ip_dir, f"generated_{DataFiles.Ip.name.lower()}_{idx}.npy"), ip_arr)
                    np.save(
                        os.path.join(seismic_dir, f"generated_{DataFiles.SEISMIC.name.lower()}_{idx}.npy"),
                        syn_seismic,
                    )
        all_mi.append(mi.cpu())
        print(f"    [{device}] {off + chunk}/{how_many}")

    return all_facies, all_ip, all_seismic, torch.cat(all_mi)


def generate_variant(
    model_path: str,
    opts: TrainingOptions,
    how_many: int,
    device: torch.device,
    wells_pyramid: tuple[torch.Tensor, ...],
    seismic_pyramid: tuple[torch.Tensor, ...],
    noise_channels: int,
    gen_output: str,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], torch.Tensor]:
    """Load a trained model and generate facies (and rock physics) samples.

    Returns
    -------
    tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], torch.Tensor]
        ``(facies, ip, seismic, mask_indexes)``.
    """
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if num_gpus >= 2:
        from concurrent.futures import Future, ThreadPoolExecutor

        # Pre-initialize CUDA contexts on every GPU from the main thread
        # so worker threads don't hit "operation not permitted" errors.
        for gpu_id in range(num_gpus):
            torch.cuda.init()
            torch.zeros(1, device=torch.device(f"cuda:{gpu_id}"))

        chunk_per_gpu = how_many // num_gpus
        remainder = how_many % num_gpus
        futures: list[
            Future[
                tuple[
                    list[np.ndarray], list[np.ndarray], list[np.ndarray], torch.Tensor
                ]
            ]
        ] = []
        with ThreadPoolExecutor(max_workers=num_gpus) as pool:
            start = 0
            for gpu_id in range(num_gpus):
                count = chunk_per_gpu + (1 if gpu_id < remainder else 0)
                dev = torch.device(f"cuda:{gpu_id}")
                futures.append(
                    pool.submit(
                        _generate_on_device,
                        dev,
                        model_path,
                        opts,
                        count,
                        wells_pyramid,
                        seismic_pyramid,
                        noise_channels,
                        gen_output,
                        start,
                    )
                )
                start += count
        facies: list[np.ndarray] = []
        ip: list[np.ndarray] = []
        seismic: list[np.ndarray] = []
        mask_indexes_list: list[torch.Tensor] = []
        for fut in futures:
            f, i_arr, s_arr, mi = fut.result()
            facies.extend(f)
            ip.extend(i_arr)
            seismic.extend(s_arr)
            mask_indexes_list.append(mi)

        return facies, ip, seismic, torch.cat(mask_indexes_list)
    else:
        return _generate_on_device(
            device,
            model_path,
            opts,
            how_many,
            wells_pyramid,
            seismic_pyramid,
            noise_channels,
            gen_output,
            0,
        )
