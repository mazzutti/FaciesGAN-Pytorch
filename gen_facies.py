"""Utilities and CLI for generating facies samples and comparison plots.

This module contains helpers to generate samples from a trained
FaciesGAN model, create comparison plots between real and generated
facies, and provide MDS, UMAP, Isomap, and t-SNE visualizations. It is primarily a
convenience script used offline after training.
"""

import json
import os
import random
import time
import warnings
from argparse import ArgumentParser

import numpy as np
import scipy.sparse as sparse  # type: ignore
import tifffile as tif
import torch
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.markers import MarkerStyle
from numpy.typing import NDArray
from sklearn.manifold import MDS, TSNE, Isomap  # type: ignore
from sklearn.metrics import euclidean_distances  # type: ignore
from umap import UMAP  # type: ignore

import utils
from background_workers import submit_plot_generated_outputs
from config import OPT_FILE
from datasets.dataset import TorchPyramidsDataset
from datasets.utils import build_conditioning_pyramids
from log import format_time
from models.facies_gan import TorchFaciesGAN
from models.utils import calculate_noise_channels
from options import TrainingOptions


def generate_facies(
    model: TorchFaciesGAN,
    how_many: int,
    model_path: str,
    options: TrainingOptions,
    dataset: TorchPyramidsDataset | None = None,
    *,
    wells_pyramid: dict[int, torch.Tensor] | None = None,
    seismic_pyramid: dict[int, torch.Tensor] | None = None,
) -> tuple[list[NDArray[np.float32]], torch.Tensor]:
    """Generate facies realizations using trained FaciesGAN model.

    Loads model weights, generates noise with well conditioning, and produces
    synthetic facies images.

    Parameters
    ----------
    model : TorchFaciesGAN
        FaciesGAN model instance.
    how_many : int
        Number of facies realizations to generate.
    model_path : str
        Path to directory containing trained model checkpoints.
    options : TrainingOptions
        Generation options including wells, rec flag, etc.
    dataset : TorchPyramidsDataset | None
        Dataset providing wells/seismic conditioning pyramids.  Can be
        omitted when *wells_pyramid* and *seismic_pyramid* are supplied
        directly (avoids the expensive facies-pyramid construction).
        Pre-built wells conditioning dict keyed by scale index.
        Pre-built seismic conditioning dict keyed by scale index.

    Returns
    -------
    tuple[list[np.ndarray], torch.Tensor]
        Tuple of (generated_facies, mask_indexes) where generated_facies is
        a list of NumPy arrays and mask_indexes are the well conditioning
        indices used.
    """
    model.load(model_path, load_discriminator=False, load_wells=False)

    mask_indexes = torch.tensor(
        [random.choice(options.wells) for _ in range(how_many)], device=model.device
    )

    # Get the highest scale (finest resolution)
    max_scale = len(model.noise_amps) - 1

    # Build conditioning pyramids from dataset (if not supplied directly)
    if wells_pyramid is None or seismic_pyramid is None:
        assert (
            dataset is not None
        ), "Either dataset or both wells_pyramid/seismic_pyramid must be provided"
        wells_pyramid, seismic_pyramid = build_conditioning_pyramids(options)

    # Get the highest scale (finest resolution)
    max_scale = len(model.noise_amps) - 1

    # Generate noise for the maximum scale
    noises = model.get_pyramid_noise(
        max_scale, mask_indexes, wells_pyramid, seismic_pyramid, rec=options.rec
    )

    with torch.no_grad():
        generated_facies: list[NDArray[np.float32]] = [
            utils.torch2np(gen_facie.unsqueeze(0), denormalize=True)
            for gen_facie in model.generator(
                noises, model.get_noise_amplitude(max_scale)
            )
        ]
    return generated_facies, mask_indexes


def generate_comparison_plots(
    model: TorchFaciesGAN,
    dataset: TorchPyramidsDataset,
    model_path: str,
    out_path: str,
    options: TrainingOptions,
    num_generated: int = 3,
    num_real: int = 5,
    scale: int | None = None,
) -> None:
    """Generate facies comparison plots showing real vs generated facies.

    Creates visual comparisons in a grid layout with real facies and
    multiple generated variants per real sample.

    Parameters
    ----------
    model : TorchFaciesGAN
        Trained FaciesGAN model instance.
    dataset : TorchPyramidsDataset
        Dataset containing facies and wells pyramids.
    model_path : str
        Path to directory containing trained model checkpoints.
    out_path : str
        Directory to save the comparison plots.
    options : TrainingOptions
        Training options (needed for use_wells / use_seismic flags).
    num_generated : int, optional
        Number of generated variants per real facies. Defaults to 3.
    num_real : int, optional
        Number of real facies rows per plot. Defaults to 5.
    scale : int | None, optional
        Pyramid scale to use. If None, uses the finest scale. Defaults to None.
    """
    model.load(model_path, load_discriminator=False, load_wells=False)

    # Use finest scale if not specified
    if scale is None:
        scale = len(model.noise_amps) - 1

    # Build conditioning pyramids from dataset
    wells_pyramid, seismic_pyramid = build_conditioning_pyramids(options)

    facies_scale, wells_scale, _ = dataset.get_scale_data(scale)
    num_images = facies_scale.shape[0]

    print(f"Generating comparison plots for {num_images} facies at scale {scale}...")

    # Process images in batches
    for start in range(0, num_images, num_real):
        end = min(start + num_real, num_images)
        if end - start < num_real:
            # Skip incomplete batch for consistent plotting
            break

        real = facies_scale[start:end]

        # Build masks from wells
        if wells_scale.numel() > 0:
            wells = wells_scale[start:end]
            masks = (wells.abs().sum(dim=1, keepdim=True) > 0).float()
        else:
            masks = None

        # Generate fake samples using the trained model
        fake_list: list[torch.Tensor] = []
        for i_idx in range(start, end):
            noises = model.get_pyramid_noise(
                scale,
                torch.tensor([i_idx] * num_generated, device=model.device),
                wells_pyramid,
                seismic_pyramid,
            )
            with torch.no_grad():
                fake = model.generator(
                    noises,
                    model.get_noise_amplitude(scale),
                    stop_scale=scale,
                )
                fake_list.append(fake.detach().cpu())

        submit_plot_generated_outputs(
            torch.stack(fake_list), real, scale, start, out_path, masks
        )

        print(
            f"Submitted async plot job for indices {start}..{end-1} -> {out_path}/gen_{scale}_{start}.png"
        )


def plot_mds(
    fake_facies: list[NDArray[np.float32]],
    mask_indexes: torch.Tensor,
    options: TrainingOptions,
    dataset: TorchPyramidsDataset,
    save_path: str | None = None,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Plot MDS embedding comparing real and generated facies.

    Parameters
    ----------
    fake_facies : list[np.ndarray]
        List of generated facies arrays (flattenable) to include in the plot.
    mask_indexes : torch.Tensor
        Indices used for well-based conditioning corresponding to the generated
        facies.
    options : TrainingOptions
        Options containing dataset indices and other generation flags.
    dataset : TorchPyramidsDataset
        Dataset providing real facies data for comparison.
    save_path : str | None, optional
        If provided, save the plot to this file path instead of showing
        it interactively.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(real_reduced, fake_reduced)`` — full (un-subsetted) real MDS
        embedding and fake MDS embedding, so callers can reuse them.
    """
    # fake_facies parameter is a list of numpy arrays. Do not rebind the parameter
    # to an ndarray (which would violate the annotated type). Use a new local
    # variable for the stacked ndarray representation.
    fake_facies_arr = np.stack(fake_facies, 0)
    if fake_facies_arr.shape[-1] == 1:
        fake_facies_arr = fake_facies_arr.squeeze(-1)
    real_facies, _, _ = dataset.get_scale_data(-1)
    real_facies = np.reshape(
        utils.torch2np(real_facies, denormalize=True),
        [real_facies.shape[0], -1],
    )
    fake_facies_arr = np.reshape(fake_facies_arr, [len(mask_indexes), -1])

    real_facies_similarities = euclidean_distances(real_facies)
    fake_facies_similarities = euclidean_distances(fake_facies_arr)
    mds = MDS(
        n_components=2,
        max_iter=3000,
        eps=1e-9,
        random_state=np.random.RandomState(seed=3),
        metric="precomputed",
        n_init=4,
        init="random",  # type: ignore
        n_jobs=1,
        normalized_stress="auto",
    )
    real_facies_similarities = np.multiply(
        real_facies_similarities + real_facies_similarities.T, 0.5
    )
    fake_facies_similarities = np.multiply(
        fake_facies_similarities + fake_facies_similarities.T, 0.5
    )
    real_facies_reduced = mds.fit(real_facies_similarities).embedding_
    fake_facies_reduced = mds.fit(fake_facies_similarities).embedding_
    plt.scatter(real_facies_reduced[options.wells, 0], real_facies_reduced[options.wells, 1])  # type: ignore
    plt.scatter(fake_facies_reduced[:, 0], fake_facies_reduced[:, 1])  # type: ignore
    plt.title("MDS Visualization of FaciesGAN generated facies")  # type: ignore
    plt.xlabel("MDS Dimension 1")  # type: ignore
    plt.ylabel("MDS Dimension 2")  # type: ignore
    plt.legend(("Real Facies", "Fake Facies"), loc="upper right")  # type: ignore
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")  # type: ignore
        plt.close()  # type: ignore
    else:
        plt.show()  # type: ignore
    return real_facies_reduced, fake_facies_reduced


def plot_umap(
    fake_facies: list[NDArray[np.float32]],
    mask_indexes: torch.Tensor,
    options: TrainingOptions,
    dataset: TorchPyramidsDataset,
    save_path: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Plot UMAP embedding comparing real and generated facies.

    Parameters
    ----------
    fake_facies : list[np.ndarray]
        List of generated facies arrays (flattenable) to include in the plot.
    mask_indexes : torch.Tensor
        Indices used for well-based conditioning corresponding to the generated
        facies.
    options : TrainingOptions
        Options containing dataset indices and other generation flags.
    dataset : TorchPyramidsDataset
        Dataset providing real facies data for comparison.
    save_path : str | None, optional
        If provided, save the plot to this file path instead of showing
        it interactively.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(real_reduced, fake_reduced)`` — full (un-subsetted) real UMAP
        embedding and fake UMAP embedding, so callers can reuse them.
    """
    fake_facies_arr = np.stack(fake_facies, 0)
    if fake_facies_arr.shape[-1] == 1:
        fake_facies_arr = fake_facies_arr.squeeze(-1)
    real_facies, _, _ = dataset.get_scale_data(-1)
    real_facies_flat = np.reshape(
        utils.torch2np(real_facies, denormalize=True),
        [real_facies.shape[0], -1],
    )
    fake_facies_flat = np.reshape(fake_facies_arr, [len(mask_indexes), -1])

    # Fit UMAP on the combined real + fake data so both share the same space
    combined = np.concatenate([real_facies_flat, fake_facies_flat], axis=0).astype(
        np.float32
    )
    reducer = UMAP(  # type: ignore
        n_components=2,
        n_neighbors=min(15, combined.shape[0] - 1),
        min_dist=0.1,
        metric="euclidean",
        random_state=42,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=sparse.SparseEfficiencyWarning)  # type: ignore
        warnings.filterwarnings("ignore", category=UserWarning, module="umap")
        embedding: np.ndarray = reducer.fit_transform(combined)  # type: ignore

    n_real = real_facies_flat.shape[0]
    real_reduced: np.ndarray = embedding[:n_real]  # type: ignore
    fake_reduced: np.ndarray = embedding[n_real:]  # type: ignore

    plt.scatter(real_reduced[options.wells, 0], real_reduced[options.wells, 1])  # type: ignore
    plt.scatter(fake_reduced[:, 0], fake_reduced[:, 1])  # type: ignore
    plt.title("UMAP Visualization of FaciesGAN generated facies")  # type: ignore
    plt.xlabel("UMAP Dimension 1")  # type: ignore
    plt.ylabel("UMAP Dimension 2")  # type: ignore
    plt.legend(("Real Facies", "Fake Facies"), loc="upper right")  # type: ignore
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")  # type: ignore
        plt.close()  # type: ignore
    else:
        plt.show()  # type: ignore
    return real_reduced, fake_reduced  # type: ignore


def plot_isomap(
    fake_facies: list[NDArray[np.float32]],
    mask_indexes: torch.Tensor,
    options: TrainingOptions,
    dataset: TorchPyramidsDataset,
    save_path: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Plot Isomap embedding comparing real and generated facies.

    Parameters
    ----------
    fake_facies : list[np.ndarray]
        List of generated facies arrays (flattenable) to include in the plot.
    mask_indexes : torch.Tensor
        Indices used for well-based conditioning corresponding to the generated
        facies.
    options : TrainingOptions
        Options containing dataset indices and other generation flags.
    dataset : TorchPyramidsDataset
        Dataset providing real facies data for comparison.
    save_path : str | None, optional
        If provided, save the plot to this file path instead of showing
        it interactively.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(real_reduced, fake_reduced)`` — full (un-subsetted) real Isomap
        embedding and fake Isomap embedding, so callers can reuse them.
    """
    fake_facies_arr = np.stack(fake_facies, 0)
    if fake_facies_arr.shape[-1] == 1:
        fake_facies_arr = fake_facies_arr.squeeze(-1)
    real_facies, _, _ = dataset.get_scale_data(-1)
    real_facies_flat = np.reshape(
        utils.torch2np(real_facies, denormalize=True),
        [real_facies.shape[0], -1],
    )
    fake_facies_flat = np.reshape(fake_facies_arr, [len(mask_indexes), -1])

    # Fit Isomap on the combined real + fake data so both share the same space
    combined = np.concatenate([real_facies_flat, fake_facies_flat], axis=0).astype(
        np.float32
    )
    n_neighbors = min(5, combined.shape[0] - 1)
    iso = Isomap(n_components=2, n_neighbors=n_neighbors)
    embedding: np.ndarray = iso.fit_transform(combined)  # type: ignore[assignment]

    n_real = real_facies_flat.shape[0]
    real_reduced: np.ndarray = embedding[:n_real]
    fake_reduced: np.ndarray = embedding[n_real:]

    plt.scatter(real_reduced[options.wells, 0], real_reduced[options.wells, 1])  # type: ignore
    plt.scatter(fake_reduced[:, 0], fake_reduced[:, 1])  # type: ignore
    plt.title("Isomap Visualization of FaciesGAN generated facies")  # type: ignore
    plt.xlabel("Isomap Dimension 1")  # type: ignore
    plt.ylabel("Isomap Dimension 2")  # type: ignore
    plt.legend(("Real Facies", "Fake Facies"), loc="upper right")  # type: ignore
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")  # type: ignore
        plt.close()  # type: ignore
    else:
        plt.show()  # type: ignore
    return real_reduced, fake_reduced


def plot_tsne(
    fake_facies: list[NDArray[np.float32]],
    mask_indexes: torch.Tensor,
    options: TrainingOptions,
    dataset: TorchPyramidsDataset,
    save_path: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Plot t-SNE embedding comparing real and generated facies.

    Parameters
    ----------
    fake_facies : list[np.ndarray]
        List of generated facies arrays (flattenable) to include in the plot.
    mask_indexes : torch.Tensor
        Indices used for well-based conditioning corresponding to the generated
        facies.
    options : TrainingOptions
        Options containing dataset indices and other generation flags.
    dataset : TorchPyramidsDataset
        Dataset providing real facies data for comparison.
    save_path : str | None, optional
        If provided, save the plot to this file path instead of showing
        it interactively.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(real_reduced, fake_reduced)`` \u2014 full (un-subsetted) real t-SNE
        embedding and fake t-SNE embedding, so callers can reuse them.
    """
    fake_facies_arr = np.stack(fake_facies, 0)
    if fake_facies_arr.shape[-1] == 1:
        fake_facies_arr = fake_facies_arr.squeeze(-1)
    real_facies, _, _ = dataset.get_scale_data(-1)
    real_facies_flat = np.reshape(
        utils.torch2np(real_facies, denormalize=True),
        [real_facies.shape[0], -1],
    )
    fake_facies_flat = np.reshape(fake_facies_arr, [len(mask_indexes), -1])

    # Fit t-SNE on the combined real + fake data so both share the same space
    combined = np.concatenate([real_facies_flat, fake_facies_flat], axis=0).astype(
        np.float32
    )
    tsne = TSNE(
        n_components=2,
        perplexity=min(30.0, (combined.shape[0] - 1) / 3.0),
        random_state=42,
    )
    embedding: np.ndarray = tsne.fit_transform(combined)  # type: ignore[assignment]

    n_real = real_facies_flat.shape[0]
    real_reduced: np.ndarray = embedding[:n_real]
    fake_reduced: np.ndarray = embedding[n_real:]

    plt.scatter(real_reduced[options.wells, 0], real_reduced[options.wells, 1])  # type: ignore
    plt.scatter(fake_reduced[:, 0], fake_reduced[:, 1])  # type: ignore
    plt.title("t-SNE Visualization of FaciesGAN generated facies")  # type: ignore
    plt.xlabel("t-SNE Dimension 1")  # type: ignore
    plt.ylabel("t-SNE Dimension 2")  # type: ignore
    plt.legend(("Real Facies", "Fake Facies"), loc="upper right")  # type: ignore
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")  # type: ignore
        plt.close()  # type: ignore
    else:
        plt.show()  # type: ignore
    return real_reduced, fake_reduced


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--how_many", help="how many facies", type=int, required=True)
    parser.add_argument("--model_path", help="models path", type=str, required=True)
    parser.add_argument("--out_path", help="path to save the generated facie", type=str)
    parser.add_argument("--use_gpu", help="use available GPU", action="store_true")
    parser.add_argument(
        "--plot_mds", help="plot the multi dimensional scaling", action="store_true"
    )
    parser.add_argument(
        "--plot_umap", help="plot the UMAP embedding", action="store_true"
    )
    parser.add_argument(
        "--plot_isomap", help="plot the Isomap embedding", action="store_true"
    )
    parser.add_argument(
        "--plot_tsne", help="plot the t-SNE embedding", action="store_true"
    )
    parser.add_argument(
        "--wells",
        help="list of well indices to generate facies from",
        type=int,
        nargs="+",
        default=tuple(range(200)),
    )
    parser.add_argument(
        "--rec",
        help=(
            "generate a sample with the reconstruction noise. "
            "The reconstruction sample will have the same size as the TI"
        ),
        action="store_true",
    )
    parser.add_argument("--gpu_device", help="GPU device", type=int, default=0)
    parser.add_argument(
        "--plot_well_mask",
        help="Add/plot also the well masks on each generated facies",
        action="store_true",
    )
    parser.add_argument(
        "--comparison_plots",
        help="Generate comparison plots showing real vs generated facies",
        action="store_true",
    )
    parser.add_argument(
        "--num_generated",
        help="Number of generated variants per real facies (for comparison plots)",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--num_real",
        help="Number of real facies rows per comparison plot",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--plot_scale",
        help="Pyramid scale to use for comparison plots (default: finest scale)",
        type=int,
        default=None,
    )

    arguments = parser.parse_args()

    if arguments.out_path is None:
        arguments.out_path = arguments.model_path

    # Place generated images in a dedicated subdirectory
    gen_output = os.path.join(arguments.out_path, "generated")
    os.makedirs(gen_output, exist_ok=True)

    device = utils.resolve_device(arguments.gpu_device)

    with open(os.path.join(arguments.model_path, OPT_FILE), "r") as f:
        json_data = json.load(f)

    # Build a proper TrainingOptions from the saved JSON and CLI overrides,
    # filtering out keys not accepted by TrainingOptions (e.g. output_fullpath).
    import inspect

    _valid_keys = set(inspect.signature(TrainingOptions.__init__).parameters) - {"self"}
    args = TrainingOptions(**{k: v for k, v in json_data.items() if k in _valid_keys})
    args.rec = arguments.rec
    args.wells = arguments.wells
    args.device = device
    args.compile_backend = False  # no need to compile for one-shot generation

    start_time = time.time()

    print("Generating facies...")

    dataset: TorchPyramidsDataset = TorchPyramidsDataset(args)

    masked_facies: list[torch.Tensor] = []
    for i in range(len(dataset)):
        facies_s, wells_s, _ = dataset.get_scale_data(i)
        if wells_s.numel() > 0:
            masks_s = (wells_s.abs().sum(dim=1, keepdim=True) > 0).float()
            masked_facies.append(facies_s * masks_s)
        else:
            masked_facies.append(facies_s)

    noise_ch = calculate_noise_channels(options=args)
    faciesGAN = TorchFaciesGAN(options=args, device=device, noise_channels=noise_ch)

    if arguments.comparison_plots:
        # Generate comparison plots instead of individual facies
        comparison_out = os.path.join(gen_output, "comparison_plots")
        os.makedirs(comparison_out, exist_ok=True)
        generate_comparison_plots(
            faciesGAN,
            dataset,
            arguments.model_path,
            comparison_out,
            options=args,
            num_generated=arguments.num_generated,
            num_real=arguments.num_real,
            scale=arguments.plot_scale,
        )
        print(f"Comparison plots saved to '{comparison_out}'.")
        print(f"Total time: {format_time(int(time.time() - start_time))}")
        exit(0)

    facies, mi = generate_facies(
        faciesGAN, arguments.how_many, arguments.model_path, args, dataset
    )

    if arguments.plot_mds:
        plot_mds(facies, mi, args, dataset)
    if arguments.plot_umap:
        plot_umap(facies, mi, args, dataset)
    if arguments.plot_isomap:
        plot_isomap(facies, mi, args, dataset)
    if arguments.plot_tsne:
        plot_tsne(facies, mi, args, dataset)
    if arguments.plot_well_mask:
        for i, (facie, masked_facie) in enumerate(
            zip(facies, [masked_facies[-1][i] for i in mi]), 1
        ):
            masked_facie_arr = np.squeeze(masked_facie.numpy())
            mask_index = np.argmax(np.sum(np.squeeze(masked_facie) != 0, axis=0))
            # create figure and axes for plotting
            fig: Figure
            axes: Axes
            fig, axes = plt.subplots(1, 1)  # type: ignore
            # Convert categorical facies to RGB for visualization
            facie_rgb = utils.facies_to_rgb(facie.squeeze())
            # facies_to_rgb returns (3, H, W), matplotlib expects (H, W, 3)
            axes.imshow(np.transpose(facie_rgb, (1, 2, 0)))  # type: ignore
            # Ensure numeric dtypes for matplotlib scatter to avoid analyzer/runtime issues
            x_coords = np.full(masked_facie_arr.shape[0], mask_index, dtype=float)
            y_coords = np.arange(masked_facie_arr.shape[0], dtype=float)
            colors = (masked_facie_arr[:, mask_index] >= 0.5).astype(np.int8)
            axes.scatter(  # type: ignore
                x_coords,
                y_coords,
                c=colors,
                s=1,
                marker=MarkerStyle("s"),
                cmap="plasma",
                label="Facies Mask",
            )
            axes.set_xticks([])  # type: ignore
            axes.set_yticks([])  # type: ignore
            axes.axis("off")  # type: ignore

            out_file = os.path.join(gen_output, f"generated_facie_{i}.tif")
            fig.savefig(out_file)  # type: ignore
            plt.close(fig)
    else:
        for i, facie in enumerate(facies, 1):
            tif.imwrite(os.path.join(gen_output, f"generated_facie_{i}.tif"), facie)  # type: ignore

    generated_pattern = os.path.join(gen_output, "generated_facie_[1, 2, ...].tif")
    print(f"Facies generated at '{generated_pattern}'.")
    print(f"Total time: {format_time(int(time.time() - start_time))}")
