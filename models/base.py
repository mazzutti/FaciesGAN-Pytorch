import math
import os
from abc import ABC, abstractmethod
from typing import Any, cast

import torch

from config import AMP_FILE, D_FILE, G_FILE, M_FILE, SHAPE_FILE
from metrics import (
    DiscriminatorMetrics,
    GeneratorMetrics,
    IterableMetrics,
    ScaleMetrics,
)
from models.discriminator import Discriminator
from models.generator import Generator
from models.utils import calculate_channels
from options import TrainingOptions


class FaciesGAN(ABC):
    """Base class for FaciesGAN.

    Responsibilities:
    - Store training/configuration parameters from `TrainingOptions`.
    - Provide core GAN utilities (scale bookkeeping, feature
        calculation, orchestration of scale initialization).
    - Provide generic checkpoint traversal (`save_scale` / `load`) while
        delegating actual serialization to concrete subclasses via hooks.

    Subclass contract (short):
    - Implement `build_generator`, `create_discriminators_container`, and
        `move_to_device` for framework object construction/device placement.
    - Implement I/O hooks named `save_*`, `load_*`, and presence checks
        used by the base `save_scale` / `load` orchestration.

    Example
    -------
    A concrete subclass should call:

            super().__init__(options, wells, seismic, noise_channels)
            self.setup_framework(device)

    After that the base will manage scale-level orchestration while the
    subclass constructs modules and performs framework-specific ops.

    """

    # The framework-specific generator instance
    generator: Generator

    # The framework-specific discriminator instance
    discriminator: Discriminator

    # Padding size for noise tensors
    zero_padding: int

    def __init__(
        self,
        options: TrainingOptions,
        noise_channels: int = 4,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Initialize the FaciesGAN base with training options.

        Parameters
        ----------
        options : TrainingOptions
            Training options containing hyperparameters and configuration.
        noise_channels : int, optional
            Number of noise channels for the generator input (default is 3).

        args : Any
            Additional positional arguments (not used).
        kwargs : Any
            Additional keyword arguments (not used).
        """

        # Store options so framework subclasses can access flags at runtime
        self.options = options

        # Basic training / architecture parameters
        self.num_parallel_scales = options.num_parallel_scales

        # Centralize channel bookkeeping
        counts = calculate_channels(options)
        self.num_facies_classes = counts["facies"]
        self.total_output_channels = counts["generator_out"]

        # Architecture channel dimensions
        self.disc_input_channels = self.total_output_channels
        self.gen_input_channels = noise_channels
        self.gen_output_channels = self.total_output_channels
        self.num_noise_channels = max(
            options.noise_channels, self.total_output_channels
        )
        self.base_channel = self.total_output_channels

        # training hyperparameters
        self.discriminator_steps = options.discriminator_steps
        self.scale0_disc_steps_multiplier = getattr(
            options, "scale0_disc_steps_multiplier", 1
        )
        self.scale0_loss_multiplier = getattr(options, "scale0_loss_multiplier", 1.0)

        # generator hyperparameters
        self.generator_steps = options.generator_steps

        # loss weights
        self.gradient_loss_penalty = options.gradient_loss_penalty

        # other loss/configuration params
        self.facies_rec_loss_penalty = options.facies_rec_loss_penalty

        # well/mask loss weight
        self.well_loss_penalty = options.well_loss_penalty

        # diversity loss params
        self.diversity_loss_penalty = getattr(options, "diversity_loss_penalty", 1.0)

        # adversarial loss penalty
        self.adversarial_loss_penalty = getattr(
            options, "adversarial_loss_penalty", 1.0
        )

        # number of diversity samples
        self.num_diversity_samples = options.num_diversity_samples

        # network sizing params
        self.num_feature = options.num_feature

        # minimum number of features
        self.min_num_feature = options.min_num_feature

        # network architecture params
        self.num_layer = options.num_layer

        # kernel size
        self.kernel_size = options.kernel_size

        # padding size
        self.padding_size = options.padding_size

        # pyramid scales
        self.shapes: list[tuple[int, ...]] = []

        # noise/reconstruction data
        self.rec_noise: list[Any] = []

        # noise amplitudes
        self.noise_amps: list[torch.Tensor] = []

        # active scales set
        self.active_scales: set[int] = set()

        # Lazy gradient penalty: compute GP every N discriminator steps
        # to amortise the cost of create_graph=True double backward.
        # The GP weight is multiplied by gp_interval to compensate.
        self.gp_interval: int = getattr(options, "gp_interval", 8)
        self._disc_step_counter: int = 0

        # Per-scale loss normalization factors: track discriminator output
        # magnitude to normalize losses across scales. Without normalization,
        # coarse scales (e.g., scale 0) produce much larger D outputs and
        # dominate training. Uses exponential moving average for stability.
        self.loss_scale_factors: dict[int, float] = {}

        # EMA decay for loss scale factor updates (0.99 = slow update)
        self.loss_scale_ema_decay: float = getattr(
            options, "loss_scale_ema_decay", 0.99
        )

    def update_loss_scale_factor(self, scale: int, d_mag: float) -> None:
        """Update the EMA loss scale factor for a given scale.

        Called after discriminator forward (outside autodiff) to track
        discriminator output magnitude per scale.

        Parameters
        ----------
        scale : int
            Pyramid scale index.
        d_mag : float
            Discriminator output magnitude (|real_loss| + |fake_loss|).
        """
        if scale not in self.loss_scale_factors:
            self.loss_scale_factors[scale] = d_mag if d_mag > 0 else 1.0
        else:
            decay = self.loss_scale_ema_decay
            self.loss_scale_factors[scale] = (
                decay * self.loss_scale_factors[scale] + (1 - decay) * d_mag
            )

    def get_loss_scale_factor(self, scale: int) -> float:
        """Return the current loss normalization factor for a scale.

        Returns 1.0 if no factor has been recorded yet (first iteration).
        """
        return max(self.loss_scale_factors.get(scale, 1.0), 1e-4)

    @abstractmethod
    def __call__(
        self, *args: Any, **kwds: Any
    ) -> ScaleMetrics | tuple[IterableMetrics, ...]:
        """Framework-specific forward method for training step.

        Parameters
        ----------
        args : Any
            Positional arguments for the forward call.
        kwds : Any
            Keyword arguments for the forward call.
        Returns
        -------
        ScaleMetrics[Any]
            Computed scale metrics from the forward pass.
        """
        raise NotImplementedError("Subclasses must implement __call__")

    @abstractmethod
    def build_discriminator(self) -> Discriminator:
        """Construct and return a framework-specific discriminator object.

        Concrete subclasses must implement this factory to create the
        discriminator instance (but should not move it to any device here).

        Raises
        ------
        NotImplementedError
            If the subclass does not override this method.

        Returns
        -------
        Discriminator[Any, Any]
            Constructed discriminator instance.
        """
        raise NotImplementedError("Subclasses must implement build_discriminator")

    @abstractmethod
    def build_generator(self) -> Generator:
        """Construct and return a framework-specific generator object.

        Concrete subclasses must implement this factory to create the
        generator instance (but should not move it to any device here).

        Raises
        ------
        NotImplementedError
            If the subclass does not override this method.

        Returns
        -------
        Generator[Any, Any]
            Constructed generator instance.
        """
        raise NotImplementedError("Subclasses must implement build_generator")

    @abstractmethod
    def concatenate_tensors(self, tensors: list[Any]) -> Any:
        """Concatenate a list of tensors along a specified dimension.

        Parameters
        ----------
        tensors : list[Any]
            List of tensors to concatenate.
        Returns
        -------
        Any
            Concatenated tensor.

        Raises
        ------
        NotImplementedError
            If the subclass does not override this method.
        """
        raise NotImplementedError("Subclasses must implement concatenate_tensors")

    @abstractmethod
    def split_tensor(self, tensor: Any, chunks: int) -> list[Any]:
        """Split a tensor into ``chunks`` equal parts along the batch dimension.

        Parameters
        ----------
        tensor : Any
            Tensor to split (batch dimension is dim 0).
        chunks : int
            Number of equal-sized chunks.

        Returns
        -------
        list[Any]
            List of ``chunks`` tensors.

        Raises
        ------
        NotImplementedError
            If the subclass does not override this method.
        """
        raise NotImplementedError("Subclasses must implement split_tensor")

    @abstractmethod
    def cat_batch(self, tensors: list[Any]) -> Any:
        """Concatenate tensors along the batch (first) dimension.

        Parameters
        ----------
        tensors : list[Any]
            List of tensors to concatenate along dim 0.

        Returns
        -------
        Any
            Concatenated tensor.

        Raises
        ------
        NotImplementedError
            If the subclass does not override this method.
        """
        raise NotImplementedError("Subclasses must implement cat_batch")

    @abstractmethod
    def compute_diversity_loss(self, fake_samples: list[Any]) -> Any:
        """Return diversity loss tensor computed from `fake_samples`.

        Parameters
        ----------
        fake_samples : list[Any]
            Generated tensor samples used to compute diversity loss.

        Returns
        -------
        Any
            Diversity loss tensor.

        Raises
        ------
        NotImplementedError
            If the subclass does not override this method.
        """
        raise NotImplementedError("Subclasses must implement compute_diversity_loss")

    @abstractmethod
    def compute_gradient_penalty(self, scale: int, real: Any, fake: Any) -> Any:
        """Return gradient penalty tensor for `scale` computed by subclass.

        Parameters
        ----------
        scale : int
            Scale index for which to compute the gradient penalty.
        real : Any
            Real tensor samples for the current scale.
        fake : Any
            Generated fake tensor samples for the current scale.

        Returns
        -------
        Any
            Gradient penalty loss tensor.

        Raises
        ------
        NotImplementedError
            If the subclass does not override this method.
        """
        raise NotImplementedError("Subclasses must implement compute_gradient_penalty")

    @abstractmethod
    def compute_masked_loss(
        self,
        fake: Any,
        real: Any,
        well: Any,
        mask: Any,
    ) -> Any:
        """Return well/mask-based loss tensor for `fake` at `scale`.

        Parameters
        ----------
        fake : Any
            Generated tensor samples for the current scale.
        real : Any
            Real tensor samples for the current scale.
        well : Any
            Well-conditioning tensor for the current scale.
        mask : Any
            Well mask tensor for the current scale.

        Returns
        -------
        Any
            Masked loss tensor.

        Raises
        ------
        NotImplementedError
            If the subclass does not override this method.
        """
        raise NotImplementedError("Subclasses must implement compute_masked_loss")

    @abstractmethod
    def compute_facies_recovery_loss(
        self,
        indexes: torch.Tensor,
        scale: int,
        real: Any,
        rec_in: Any,
        wells_pyramid: dict[int, Any] = {},
        seismic_pyramid: dict[int, Any] = {},
    ) -> Any:
        """Return facies reconstruction loss tensor for the provided inputs.

        Parameters
        ----------
        indexes : list[int]
            List of batch/sample indices used to generate noise.
        scale : int
            Scale index for which to compute the facies recovery loss.
        real : Any
            Real tensor samples for the current scale.
        rec_in : Any
            Input tensor for reconstruction at the current scale.
        wells_pyramid : dict[int, Any], optional
            Well-conditioning tensor dictionary for the current scale.
        seismic_pyramid : dict[int, Any], optional
            Seismic-conditioning tensor dictionary for the current scale.
        Returns
        -------
        Any
            Facies reconstruction loss tensor.

        Raises
        ------
        NotImplementedError
            If the subclass does not override this method.
        """
        raise NotImplementedError(
            "Subclasses must implement compute_facies_recovery_loss"
        )

    @abstractmethod
    def finalize_discriminator_scale(self, scale: int) -> None:
        """Finalize discriminator scale after creation and optional device move.

        Parameters
        ----------
        scale : int
            Scale index that was just created.

        Raises
        ------
        NotImplementedError
            If the subclass does not override this method.
        """
        raise NotImplementedError(
            "Subclasses must implement finalize_discriminator_scale"
        )

    @abstractmethod
    def finalize_generator_scale(self, scale: int, reinit: bool) -> None:
        """Finalize generator scale after creation and optional device move.

        If `reinit` is True subclasses should initialize weights for the new
        block; otherwise they should copy weights from the previous block.

        Parameters
        ----------
        scale : int
            Scale index that was just created.
        reinit : bool
            Whether to reinitialize weights or copy from previous scale.

        Raises
        ------
        NotImplementedError
            If the subclass does not override this method.
        """
        raise NotImplementedError("Subclasses must implement finalize_generator_scale")

    @abstractmethod
    def generate_fake(self, noises: list[Any], scale: int) -> Any:
        """Generate fake images from `noises` without tracking gradients.

        Subclasses should implement this using their framework's no-grad
        mechanism (e.g., `torch.no_grad()`), returning the produced tensor.
        """
        raise NotImplementedError("Subclasses must implement generate_fake")

    @abstractmethod
    def generate_padding(self, z: Any, value: int = 0) -> Any:
        """Apply padding to a noise tensor.

        Parameters
        ----------
        z : Any
            Input noise tensor to pad.
        value : int, optional
            Padding fill value (default is 0).

        Returns
        -------
        Any
            Padded tensor.

        Raises
        ------
        NotImplementedError
            If the subclass does not override this method.
        """
        raise NotImplementedError("Subclasses must implement generate_padding")

    @abstractmethod
    def get_noise_shape(
        self, scale: int, use_base_channel: bool = True
    ) -> tuple[int, ...]:
        """Get the noise tensor shape for a specific scale.

        Parameters
        ----------
        scale : int
            Scale index for which to get the noise shape.

        Returns
        -------
        tuple[int, ...]
            Shape of the noise tensor for the specified scale.

        Raises
        ------
        NotImplementedError
            If the subclass does not override this method.
        """
        raise NotImplementedError("Subclasses must implement get_noise_shape")

    def get_rec_noise(self, scale: int) -> list[Any]:
        """Get the reconstruction noise tensor for a specific scale.

        Parameters
        ----------
        scale : int
            Scale index for which to get the reconstruction noise.

        Returns
        -------
        Any
            Reconstruction noise tensor for the specified scale.

        Raises
        ------
        NotImplementedError
            If the subclass does not override this method.
        """
        raise NotImplementedError("Subclasses must implement get_rec_noise")

    @abstractmethod
    def load_discriminator_state(self, scale_path: str, scale: int) -> None:
        """Load discriminator state for a given scale from disk.

        Parameters
        ----------
        scale_path : str
            Path to the scale directory containing saved discriminator state.
        scale : int
            The pyramid scale index being loaded.

        Raises
        ------
        NotImplementedError
            If the subclass does not override this method.
        """
        raise NotImplementedError("Subclasses must implement load_discriminator_state")

    @abstractmethod
    def load_amp(self, scale_path: str) -> None:
        """Load amplitude information for a scale and append to `self.noise_amps`.

        Parameters
        ----------
        scale_path : str
            Path to the scale directory containing the amplitude file.

        Raises
        ------
        NotImplementedError
            If the subclass does not override this method.
        """
        raise NotImplementedError("Subclasses must implement load_amp")

    @abstractmethod
    def load_shape(self, scale_path: str) -> None:
        """Load shape metadata for a scale and append to `self.shapes`.

        Parameters
        ----------
        scale_path : str
            Path to the scale directory containing the shape file.

        Raises
        ------
        NotImplementedError
            If the subclass does not override this method.
        """
        raise NotImplementedError("Subclasses must implement load_shape")

    @abstractmethod
    def load_wells(self, scale_path: str) -> None:
        """Load well-conditioning data for a scale and append to `self.wells`.

        Parameters
        ----------
        scale_path : str
            Path to the scale directory containing wells data.

        Raises
        ------
        NotImplementedError
            If the subclass does not override this method.
        """
        raise NotImplementedError("Subclasses must implement load_wells")

    @abstractmethod
    def load_generator_state(self, scale_path: str, scale: int) -> None:
        """Load generator state for a given scale from disk.

        Parameters
        ----------
        scale_path : str
            Path to the scale directory containing saved generator state.
        scale : int
            The pyramid scale index being loaded.

        Raises
        ------
        NotImplementedError
            If the subclass does not override this method.
        """
        raise NotImplementedError("Subclasses must implement load_generator_state")

    @abstractmethod
    def save_discriminator_state(self, scale_path: str, scale: int) -> None:
        """Save discriminator state for a given scale to disk.

        Parameters
        ----------
        scale_path : str
            Directory path where discriminator state should be saved.
        scale : int
            Pyramid scale index being saved.

        Raises
        ------
        NotImplementedError
            If the subclass does not override this method.
        """
        raise NotImplementedError("Subclasses must implement save_discriminator_state")

    @abstractmethod
    def save_generator_state(self, scale_path: str, scale: int) -> None:
        """Save generator state for a given scale to disk.

        Parameters
        ----------
        scale_path : str
            Directory path where generator state should be saved.
        scale : int
            Pyramid scale index being saved.

        Raises
        ------
        NotImplementedError
            If the subclass does not override this method.
        """
        raise NotImplementedError("Subclasses must implement save_generator_state")

    @abstractmethod
    def save_shape(self, scale_path: str, scale: int) -> None:
        """Save shape metadata for a given scale to disk.

        Parameters
        ----------
        scale_path : str
            Directory path where shape metadata should be written.
        scale : int
            Pyramid scale index being saved.

        Raises
        ------
        NotImplementedError
            If the subclass does not override this method.
        """
        raise NotImplementedError("Subclasses must implement save_shape")

    def compute_adversarial_loss(self, scale: int, fake: Any) -> Any:
        """Compute adversarial loss for a generated tensor at a scale.

        Parameters
        ----------
        scale : int
            Pyramid scale index for which to compute the loss.
        fake : Any
            Generated tensor produced by the generator for the given scale.

        Returns
        -------
        Any
            Scalar tensor equal to the negative mean score from the discriminator.
        """
        discriminator = self.discriminator.discs[scale]
        return self.adversarial_loss_penalty * (-discriminator(fake).mean())  # type: ignore

    @abstractmethod
    def compute_discriminator_metrics(
        self,
        indexes: torch.Tensor,
        scale: int,
        real: Any,
        wells_pyramid: dict[int, Any] = {},
        seismic_pyramid: dict[int, Any] = {},
    ) -> tuple[
        DiscriminatorMetrics | IterableMetrics,
        dict[str, Any] | None,
    ]:
        """Compute discriminator losses and gradient penalty for a scale.

        Parameters
        ----------
        indexes (list[int]):
            Batch/sample indices used to generate fake inputs.
        scale (int):
            Pyramid scale index for which to compute the metrics.
        real (Any):
            Ground-truth tensor for the current scale.
        wells_pyramid (dict[int, Any]):
            Wells tensors dictionary for conditioning, keyed by scale.
        seismic_pyramid (dict[int, Any]):
            Seismic tensors dictionary for conditioning, keyed by scale.
        Returns
        -------
        tuple[DiscriminatorMetrics[Any] | IterableMetrics[Any], dict[str, Any] | None]:
            Container with total, real, fake and gp losses, and optional gradients dict.

        Raises
        ------
        NotImplementedError
            If the subclass does not override this method.
        """
        raise NotImplementedError(
            "Subclasses must implement compute_discriminator_metrics"
        )

    @abstractmethod
    def compute_generator_metrics(
        self,
        indexes: torch.Tensor,
        scale: int,
        real: Any,
        facies_in_pyramid: dict[int, Any],
        wells_pyramid: dict[int, Any] = {},
        masks_pyramid: dict[int, Any] = {},
        seismic_pyramid: dict[int, Any] = {},
    ) -> tuple[
        GeneratorMetrics | IterableMetrics,
        dict[str, Any] | None,
    ]:
        """Common generator-metrics flow shared by frameworks.

        Parameters
        ----------
        indexes (list[int]):
            List of batch/sample indices used to generate noise.
        scale (int):
            Pyramid scale index for which to compute the metrics.
        real (Any):
            Ground-truth tensor for the current scale.
        facies_in_pyramid (dict[int, Any]):
            Facies reconstruction input tensors for the current scale.
        wells_pyramid (dict[int, Any]):
            Well log tensors for conditioning, keyed by scale.
        masks_pyramid (dict[int, Any]):
            Well/mask tensors for conditioning, keyed by scale.
        seismic_pyramid (dict[int, Any]):
            Seismic volume tensors for conditioning, keyed by scale.

        Returns
        -------
        GeneratorMetrics[Any] | IterableMetrics[Any]:
            Container with total, fake, facies_rec, well and div losses, and optional gradients list.

        Raises
        ------
        NotImplementedError
            If the subclass does not override this method.
        """
        raise NotImplementedError("Subclasses must implement compute_generator_metrics")

    @abstractmethod
    def update_discriminator_weights(
        self, scale: int, optimizer: Any, loss: Any, gradients: Any | None
    ) -> None:
        """Update discriminator weights using the provided optimizer and loss/gradients.

        This method encapsulates framework-specific optimization steps (e.g.,
        `loss.backward()` and `optimizer.step()` for PyTorch, or
        `optimizer.step()`).

        Parameters
        ----------
        scale : int
            Scale index.
        optimizer : Any
            Optimizer instance for the discriminator at the given scale.
        loss : Any
            Total loss tensor.
        gradients : Any | None
            Computed gradients (if applicable).

        Raises
        ------
        NotImplementedError
            If the subclass does not override this method.
        """
        raise NotImplementedError(
            "Subclasses must implement update_discriminator_weights"
        )

    @abstractmethod
    def update_generator_weights(
        self, scale: int, optimizer: Any, loss: Any, gradients: Any | None
    ) -> None:
        """Update generator weights using the provided optimizer and loss/gradients.

        Parameters
        ----------
        scale : int
            Scale index.
        optimizer : Any
            Generator optimizer for the current scale.
        loss : Any
            Total loss tensor.
        gradients : Any | None
            Computed gradients (if applicable).

        Raises
        ------
        NotImplementedError
            If the subclass does not override this method.
        """
        raise NotImplementedError("Subclasses must implement update_generator_weights")

    @abstractmethod
    def forward(
        self,
        generator_optimizers: dict[int, Any],
        discriminator_optimizers: dict[int, Any],
        indexes: torch.Tensor,
        facies_pyramid: dict[int, Any],
        rec_in_pyramid: dict[int, Any],
        wells_pyramid: dict[int, Any] = {},
        masks_pyramid: dict[int, Any] = {},
        seismic_pyramid: dict[int, Any] = {},
    ) -> ScaleMetrics | tuple[IterableMetrics, ...]:
        """Perform a forward pass for both discriminator and generator.

        Parameters
        ----------
        indexes : tuple[int, ...]
            Tuple of batch/sample indices used to generate noise.
        generator_optimizers : dict[int, Any]
            Dictionary mapping scale indices to optimizers for generator.
        discriminator_optimizers : dict[int, Any]
            Dictionary mapping scale indices to optimizers for discriminator.
        facies_pyramid : dict[int, Any]
            Dictionary mapping scale indices to ground-truth tensors.
        rec_in_pyramid : dict[int, Any]
            Dictionary mapping scale indices to reconstruction inputs.
        wells_pyramid : dict[int, Any], optional
            Dictionary mapping scale indices to well-conditioning tensors.
        masks_pyramid : dict[int, Any], optional
            Dictionary mapping scale indices to well/mask tensors.
        seismic_pyramid : dict[int, Any], optional
            Dictionary mapping scale indices to seismic-conditioning tensors.
        Returns
        -------
        ScaleMetrics[Any]
            Container with discriminator and generator metrics for the forward pass.

        Raises
        ------
        NotImplementedError
            If the subclass does not override this method.
        """
        raise NotImplementedError("Subclasses must implement forward")

    def generate_diverse_samples(
        self,
        indexes: torch.Tensor,
        scale: int,
        wells_pyramid: dict[int, Any] = {},
        seismic_pyramid: dict[int, Any] = {},
    ) -> list[Any]:
        """Generate multiple candidate outputs for `scale` using current generator.

        All diversity samples are generated in a **single batched forward
        pass** (batch dimension is tiled by ``num_diversity_samples``) to
        maximise GPU utilisation.  The output is then split back into
        individual samples for the downstream diversity loss.

        Parameters
        ----------
        indexes : tuple[int, ...]
            Tuple of batch/sample indices used to generate noise.
        scale : int
            Pyramid scale index for which to generate samples.
        wells_pyramid : dict[int, Any], optional
            Dictionary mapping scale indices to well-conditioning tensors for the current scale.
        seismic_pyramid : dict[int, Any], optional
            Dictionary mapping scale indices to seismic-conditioning tensors for the current scale.

        Returns
        -------
        list[Any]
            List of generated tensors from multiple noise realizations.
        """
        # During diversity warmup, use N=1 to avoid the expensive
        # batched forward (N*B → B) while the generator is still random.
        div_skip = getattr(self, "_div_skip_epochs", 0)
        cur_epoch = getattr(self, "_current_epoch", 0)
        N = 1 if cur_epoch < div_skip else self.num_diversity_samples
        if N <= 1:
            noises = self.get_pyramid_noise(
                scale, indexes, wells_pyramid, seismic_pyramid
            )
            amps = self.get_noise_amplitude(scale)
            return [self.generator(noises, amps, stop_scale=scale)]

        # Build N independent noise pyramids and concatenate along batch dim
        # so the generator runs a single forward with batch size B*N.
        noise_sets: list[list[Any]] = [
            self.get_pyramid_noise(scale, indexes, wells_pyramid, seismic_pyramid)
            for _ in range(N)
        ]
        batched_noises: list[Any] = [
            self.cat_batch([noise_sets[k][lvl] for k in range(N)])
            for lvl in range(scale + 1)
        ]
        amps = self.get_noise_amplitude(scale)
        batched_out = self.generator(batched_noises, amps, stop_scale=scale)

        # Split back into N individual samples along batch dim (dim 0).
        return self.split_tensor(batched_out, N)

    def get_pyramid_noise(
        self,
        scale: int,
        indexes: torch.Tensor,
        wells_pyramid: dict[int, Any] = {},
        seismic_pyramid: dict[int, Any] = {},
        rec: bool = False,
    ) -> list[Any]:
        """Generate noise tensors up to a specific pyramid scale (generic).

        Uses `NoiseSpec` and framework-provided callables supplied by the
        subclass (via `get_noise_gen_fn`, `get_pad_fn`, `get_cat_fn`) so the
        base implementation remains generic.

        Parameters
        ----------
        indexes : list[int]
            Batch/sample indices used to generate noise.
        scale : int
            Pyramid scale index up to which to generate noise tensors.
        wells_pyramid : dict[int, Any], optional
            Dictionary mapping scale indices to well-conditioning tensors for the current scale.
        seismic_pyramid : dict[int, Any], optional
            Dictionary mapping scale indices to seismic-conditioning tensors for the current scale.
        rec : bool, optional
            If True, return stored reconstruction noise instead of new noise.
            (default is False).

        Returns
        -------
        list[Any]
            List of noise tensors from scale 0 up to `scale`.
        """
        if rec:
            return self.get_rec_noise(scale)
        return [
            self.generate_noise(
                i,
                indexes,
                wells_pyramid.get(i) if wells_pyramid else None,
                seismic_pyramid.get(i) if seismic_pyramid else None,
            )
            for i in range(scale + 1)
        ]

    def get_noise_amplitude(self, scale: int) -> list[torch.Tensor]:
        """Return noise amplitude list up to a given scale (generic).
        If noise amps have not been computed, return list of 1.0s.

        Parameters
        ----------
        scale : int
            Pyramid scale index up to which to return amplitudes.

        Returns
        -------
        list[torch.Tensor`]
            List of noise amplitudes.
        """
        return (
            self.noise_amps[: scale + 1]
            if hasattr(self, "noise_amps") and len(self.noise_amps) > 0
            else [torch.tensor(1.0) for _ in range(scale + 1)]
        )

    def get_num_features(self, scale: int) -> tuple[int, int]:
        """Calculate feature counts for networks at a given scale.

        Features double every 4 scales up to a maximum of 128. This logic is
        remains generic and therefore lives in the base class.

        Parameters
        ----------
        scale : int
            Pyramid scale index for which to compute feature counts.

        Returns
        -------
        tuple[int, int]
            A 2-tuple of integers: (num_feature, min_num_feature).
        """
        num_feature = min(self.num_feature * pow(2, math.floor(scale / 4)), 128)
        min_num_feature = min(self.min_num_feature * pow(2, math.floor(scale / 4)), 128)

        return num_feature, min_num_feature

    def has_amp_file(self, scale_path: str) -> bool:
        """Return True if amplitude file exists in `scale_path`.

        Default implementation checks for the presence of `AMP_FILE` in the
        given scale directory. Subclasses can override if they store
        amplitude data differently.

        Parameters
        ----------
        scale_path : str
            Path to the scale directory to check for amplitude file.

        Returns
        -------
        bool
            True if amplitude file exists, False otherwise.
        """
        return os.path.exists(os.path.join(scale_path, AMP_FILE))

    def has_discriminator_checkpoint(self, scale_path: str) -> bool:
        """Return True if discriminator checkpoint exists in `scale_path`.

        Default implementation checks for the presence of `D_FILE` in the
        given scale directory. Subclasses can override if they store
        discriminator checkpoints differently.

        Parameters
        ----------
        scale_path : str
            Path to the scale directory to check for discriminator checkpoint.

        Returns
        -------
        bool
            True if discriminator checkpoint exists, False otherwise.
        """
        return os.path.exists(os.path.join(scale_path, D_FILE))

    def has_generator_checkpoint(self, scale_path: str) -> bool:
        """Return True if generator checkpoint exists in `scale_path`.

        Default implementation checks for the presence of `G_FILE` in the
        given scale directory. Subclasses can override if they store
        generator checkpoints differently.

        Parameters
        ----------
        scale_path : str
            Path to the scale directory to check for generator checkpoint.

        Returns
        -------
        bool
            True if generator checkpoint exists, False otherwise.
        """
        return os.path.exists(os.path.join(scale_path, G_FILE))

    def has_shape_file(self, scale_path: str) -> bool:
        """Return True if shape file exists in `scale_path`.

        Default implementation checks for the presence of `SHAPE_FILE` in
        the given scale directory. Subclasses can override if they store
        shapes differently.

        Parameters
        ----------
        scale_path : str
            Path to the scale directory to check for shape file.

        Returns
        -------
        bool
            True if shape file exists, False otherwise.
        """
        return os.path.exists(os.path.join(scale_path, SHAPE_FILE))

    def has_wells_file(self, scale_path: str) -> bool:
        """Return True if wells file exists in `scale_path`.

        Default implementation checks for the presence of `M_FILE` in the
        given scale directory. Subclasses may override if they use a
        different wells storage layout.

        Parameters
        ----------
        scale_path : str
            Path to the scale directory to check for wells file.

        Returns
        -------
        bool
            True if wells file exists, False otherwise.
        """
        return os.path.exists(os.path.join(scale_path, M_FILE))

    def init_discriminator_for_scale(self, scale: int) -> None:
        """Initialize discriminator for a new pyramid scale.

        Creates a new discriminator with appropriate feature counts. Each
        scale gets its own discriminator for parallel training.

        Parameters
        ----------
        scale : int
            Pyramid scale index to initialize.
        """
        num_feature, min_num_feature = self.get_num_features(scale)

        # Create the framework-specific discriminator scale block (subclass).
        self.discriminator.create_scale(num_feature, min_num_feature)

        # Let the subclass finalize the new discriminator block and apply weights.
        self.finalize_discriminator_scale(scale)

    def init_generator_for_scale(self, scale: int) -> None:
        """Generic generator-scale initializer.

        This base implementation computes the appropriate feature counts for
        `scale`, delegates creation and device-placement to framework hooks,
        and then asks the subclass to finalize the new block (either
        reinitializing weights or copying previous weights). Subclasses must
        implement the three hooks documented below.


        Parameters
        ----------
        scale : int
            Pyramid scale index to initialize.
        """
        num_feature, min_num_feature = self.get_num_features(scale)
        self.generator.create_scale(scale, num_feature, min_num_feature)
        prev_is_spade = self.is_spade_scale(scale - 1) if scale > 0 else False
        curr_is_spade = self.is_spade_scale(scale)
        reinit = prev_is_spade or curr_is_spade
        self.finalize_generator_scale(scale, reinit)

    def init_scales(self, start_scale: int, num_scales: int) -> None:
        """Initialize a consecutive range of scales.

        This generic implementation delegates the framework-specific work to
        the abstract methods `init_scale_generator` and
        `init_scale_discriminator` which concrete subclasses must implement.

        The active_scales set is replaced with exactly the scales being
        trained so that optimize_discriminator / optimize_generator only
        iterate the current group (previous groups' scales are frozen).

        Parameters
        ----------
        start_scale : int
            Pyramid scale index to start initializing from.
        num_scales : int
            Number of consecutive scales to initialize.
        """
        new_scales: set[int] = set()
        for scale in range(start_scale, start_scale + num_scales):
            self.init_generator_for_scale(scale)
            self.init_discriminator_for_scale(scale)
            new_scales.add(scale)
        self.active_scales = new_scales

    def freeze_generator_scales(self, active_scales: tuple[int, ...]) -> None:
        """Freeze generator blocks outside the active training set.

        Sets ``requires_grad_(False)`` on generator blocks for scales that
        are **not** in ``active_scales``.  This prevents ``backward()``
        from allocating gradient tensors on frozen parameters during the
        progressive forward pass, saving significant GPU memory.

        Active scales are unfrozen (``requires_grad_(True)``) to ensure
        they can be trained normally.

        Subclasses may override to add framework-specific cleanup.

        Parameters
        ----------
        active_scales : tuple[int, ...]
            Scales currently being trained.
        """
        active_set = set(active_scales)
        for i, gen in enumerate(self.generator.gens):
            if hasattr(gen, "requires_grad_"):
                gen.requires_grad_(i in active_set)  # type: ignore[union-attr]

    def trim_rec_noise(self, keep_up_to: int) -> None:
        """Move reconstruction noise tensors outside the active range to CPU.

        The progressive generator forward indexes noise by scale position,
        so we cannot remove entries.  Instead we move tensors for
        previously-trained scales to CPU, freeing CUDA memory while
        keeping them available if ever needed again.

        Parameters
        ----------
        keep_up_to : int
            Index of the first scale to *keep* on device.  Entries
            ``rec_noise[0 .. keep_up_to-1]`` are moved to CPU.
        """
        try:
            import torch

            for i in range(min(keep_up_to, len(self.rec_noise))):
                t = self.rec_noise[i]
                if isinstance(t, torch.Tensor) and t.is_cuda:
                    self.rec_noise[i] = t.cpu()
        except ImportError:
            pass

    def clear_stale_generator_grads(self, active_scales: tuple[int, ...]) -> None:
        """Free ``.grad`` tensors on generator parameters outside active scales.

        ``backward()`` computes gradients for every ``requires_grad``
        parameter in the computation graph, but only the current
        scale's optimizer zeroes them.  Leftover gradient tensors on
        other active scales waste GPU memory.  This method sets them to
        ``None`` to allow the CUDA allocator to reclaim the blocks.

        Parameters
        ----------
        active_scales : tuple[int, ...]
            Scales that were just trained (whose optimizers already
            called ``zero_grad``).
        """
        # After the last scale in the optimization loop, ALL active
        # scales' parameters may carry stale .grad tensors from the
        # backward passes of subsequent scales.
        for _idx, gen in enumerate(self.generator.gens):
            if not hasattr(gen, "parameters"):
                continue
            for p in gen.parameters():  # type: ignore[union-attr]
                if p.grad is not None:  # type: ignore[union-attr]
                    p.grad = None  # type: ignore[union-attr]

    def is_spade_scale(self, scale: int) -> bool:
        """Return True if `scale` uses SPADE (or other scale-specific flag).

        Default implementation looks for a `spade_scales` attribute on
        `self.generator` and checks membership; returns False when the
        attribute is missing. Subclasses may override for different
        generator implementations.

        Parameters
        ----------
        scale : int
            Pyramid scale index to check.

        Returns
        -------
        bool
            True if `scale` is a SPADE scale, False otherwise.
        """
        return scale in self.generator.spade_scales

    def load(
        self,
        path: str,
        load_shapes: bool = True,
        until_scale: int | None = None,
        load_discriminator: bool = False,
        load_wells: bool = False,
    ) -> int:
        """Load saved models and return the next starting scale.

        The base implementation walks the checkpoint directory structure and
        delegates actual model/state loading to subclass hooks.

        Parameters
        ----------
        path : str
            Root directory path where scale subdirectories are located.
        load_shapes : bool, optional
            Whether to load shape metadata for each scale (default is True).
        until_scale : int | None, optional
            If provided, load scales only up to (and including) this index.
            Default is None (load all available scales).
        load_discriminator : bool, optional
            Whether to load discriminator states for each scale
            (default is False).
        load_wells : bool, optional
            Whether to load well-conditioning data for each scale (default is False).

        Returns
        -------
        int
            The next scale index after the last successfully loaded scale.
        """
        scale = 0

        while os.path.exists(os.path.join(path, str(scale))):
            if until_scale is not None and scale > until_scale:
                break

            scale_path = os.path.join(path, str(scale))

            # Load generator if a checkpoint exists for this scale
            if self.has_generator_checkpoint(scale_path):
                self.init_generator_for_scale(scale)
                self.load_generator_state(scale_path, scale)

            if load_discriminator and self.has_discriminator_checkpoint(scale_path):
                self.init_discriminator_for_scale(scale)
                self.load_discriminator_state(scale_path, scale)

            # Load amplitude
            if self.has_amp_file(scale_path):
                self.load_amp(scale_path)

            # Load shapes
            if load_shapes and self.has_shape_file(scale_path):
                self.load_shape(scale_path)

            if load_wells and self.has_wells_file(scale_path):
                self.load_wells(scale_path)

            scale += 1

        return scale

    def move_to_device(self, obj: Any) -> Any:
        """Optional hook to move framework objects to `device`.

        Default implementation is a no-op. Framework subclasses (e.g.
        PyTorch) should override to call `.to(device)` on modules.

        Parameters
        ----------
        obj : Any
            Framework-specific object to move to device.

        Returns
        -------
        Any
            The same object, moved to the target device.
        """
        return obj

    def optimize_discriminator(
        self,
        indexes: torch.Tensor,
        optimizers: dict[int, Any],
        facies_pyramid: dict[int, Any],
        wells_pyramid: dict[int, Any] = {},
        seismic_pyramid: dict[int, Any] = {},
    ) -> tuple[DiscriminatorMetrics, ...] | IterableMetrics:
        """Discriminator optimization orchestration.

        This method zeroes gradients, delegates framework-specific forward
        computations to small abstract hooks implemented by subclasses,
        aggregates tensor losses for a single backward call, and steps the
        provided optimizers. It intentionally avoids importing heavy
        frameworks.

        Parameters
        ----------
        indexes : list[int]
            List of batch/sample indices used to generate noise.
        optimizers : dict[int, Any]
            Dictionary mapping scale indices to discriminator optimizers.
        facies_pyramid : dict[int, Any]
            Dictionary mapping scale indices to real tensor samples.
        wells_pyramid : dict[int, Any]
            Dictionary mapping scale indices to well-conditioning tensors.
        seismic_pyramid : dict[int, Any]
            Dictionary mapping scale indices to seismic-conditioning tensors.
        Returns
        -------
        tuple[DiscriminatorMetrics, ...] | IterableMetrics[Any]:
            Tuple of computed discriminator metrics for each scale.
        """
        step_metrics: list[DiscriminatorMetrics] = []

        # Sort active scales so that every DDP rank iterates in the same
        # order.  ``active_scales`` is a set whose iteration order is an
        # implementation detail of CPython; sorting removes any ambiguity
        # and guarantees that per-scale all-reduce calls are matched
        # across ranks (NCCL matches collectives by call order).
        sorted_scales = sorted(self.active_scales)

        for _ in range(self.discriminator_steps):

            # Compute metrics for this discriminator step only
            step_metrics = []
            for scale in sorted_scales:

                metrics, gradients = self.compute_discriminator_metrics(
                    indexes,
                    scale,
                    facies_pyramid[scale],
                    wells_pyramid,
                    seismic_pyramid,
                )

                metrics = cast(DiscriminatorMetrics, metrics)

                # Delegate the optimization step to subclass
                self.update_discriminator_weights(
                    scale,
                    optimizers[scale],
                    metrics.total,
                    gradients,
                )

                # Detach immediately after backward to release the
                # autograd graph node cycle rooted at metrics.total.
                if hasattr(metrics.total, "detach"):
                    metrics = DiscriminatorMetrics(  # type: ignore[misc]
                        total=metrics.total.detach(),  # type: ignore[union-attr]
                        real=metrics.real.detach(),  # type: ignore[union-attr]
                        fake=metrics.fake.detach(),  # type: ignore[union-attr]
                        gp=metrics.gp.detach(),  # type: ignore[union-attr]
                    )

                step_metrics.append(metrics)  # type: ignore[arg-type]

        return tuple(step_metrics)

    def optimize_generator(
        self,
        indexes: torch.Tensor,
        optimizers: dict[int, Any],
        facies_pyramid: dict[int, Any],
        rec_in_pyramid: dict[int, Any],
        wells_pyramid: dict[int, Any] = {},
        masks_pyramid: dict[int, Any] = {},
        seismic_pyramid: dict[int, Any] = {},
    ) -> tuple[GeneratorMetrics, ...] | IterableMetrics:
        """Generator optimization orchestration.

        This method handles zeroing grads, calling the per-scale
        computation hook, aggregating totals for backward, and stepping
        optimizers. Subclasses must implement
        `compute_generator_metrics` to return scale-level
        metrics and produced tensors.

        Parameters
        ----------
        indexes : list[int]
            List of batch/sample indices used to generate noise.
        optimizers : dict[int, Any]
            Dictionary mapping scale indices to generator optimizers.
        facies_pyramid : dict[int, Any]
            Dictionary mapping scale indices to real tensor samples.
        rec_in_pyramid : dict[int, Any]
            Dictionary mapping scale indices to reconstruction inputs.
        wells_pyramid : dict[int, Any]
            Dictionary mapping scale indices to well-conditioning tensors.
        masks_pyramid : dict[int, Any]
            Dictionary mapping scale indices to well/mask tensors.
        seismic_pyramid : dict[int, Any]
            Dictionary mapping scale indices to seismic-conditioning tensors.

        Returns
        -------
        tuple[GeneratorMetrics, ...] | IterableMetrics[Any]:
            Tuple of computed generator metrics for each scale.
        """

        step_metrics: list[GeneratorMetrics] = []

        sorted_scales = sorted(self.active_scales)

        for _ in range(self.generator_steps):

            # Compute per-scale metrics using subclass hook
            step_metrics = []
            for scale in sorted_scales:
                if scale >= len(facies_pyramid):
                    continue

                real = facies_pyramid[scale]

                # Ensure noise amplitudes have been initialized for this scale.
                # In normal training `noise_amp` is populated during noise
                # initialization; missing values indicate setup wasn't run.
                if len(self.noise_amps) < scale + 1:
                    raise RuntimeError(
                        f"noise_amp not initialized for scale {scale}. "
                        "Call the project's noise initialization before training."
                    )

                metrics, gradients = self.compute_generator_metrics(
                    indexes,
                    scale,
                    real,
                    rec_in_pyramid,
                    wells_pyramid,
                    masks_pyramid,
                    seismic_pyramid,
                )

                metrics = cast(GeneratorMetrics, metrics)

                # Delegate the optimization step to subclass
                self.update_generator_weights(
                    scale,
                    optimizers[scale],
                    metrics.total,
                    gradients,
                )

                # After backward the computation graph buffers are
                # freed, but the autograd Node objects still form a
                # reference cycle via metrics.total.grad_fn.  Detach
                # immediately so those nodes can be collected.
                if hasattr(metrics.total, "detach"):
                    metrics = GeneratorMetrics(  # type: ignore[misc]
                        total=metrics.total.detach(),  # type: ignore[union-attr]
                        fake=metrics.fake.detach(),  # type: ignore[union-attr]
                        rec=metrics.rec.detach(),  # type: ignore[union-attr]
                        well=metrics.well.detach(),  # type: ignore[union-attr]
                        div=metrics.div.detach(),  # type: ignore[union-attr]
                        imp=(getattr(metrics, "rp", None) or getattr(metrics, "imp", None) or self._zero_scalar).detach(),  # type: ignore[attr-defined]
                    )

                step_metrics.append(metrics)  # type: ignore[arg-type]

                # Free stale .grad on gen blocks that are NOT the
                # current scale.  backward() fans out gradients to
                # every requires_grad parameter in the graph (which
                # includes all active gen blocks for multi-scale
                # forward), but only the current scale's optimizer
                # zeroes its own.  Clearing immediately avoids keeping
                # those gradient tensors alive until the next scale's
                # zero_grad or end-of-epoch cleanup.
                for idx, gen in enumerate(self.generator.gens):
                    if idx == scale:
                        continue
                    if not hasattr(gen, "parameters"):
                        continue
                    for p in gen.parameters():  # type: ignore[union-attr]
                        if p.grad is not None:  # type: ignore[union-attr]
                            p.grad = None  # type: ignore[union-attr]

        return tuple(step_metrics)

    def save_amp(self, scale_path: str, scale: int) -> None:
        """Save amplitude (noise_amp) for `scale` into `scale_path` directory.

        This default implementation writes the float value of `self.noise_amps[scale]`
        to a small text file named by `AMP_FILE`. Subclasses may override if they
        need different semantics.

        Parameters
        ----------
        scale_path : str
            Directory path where amplitude file should be saved.
        scale : int
            Pyramid scale index being saved.
        """
        if scale < len(self.noise_amps):
            amp_path = os.path.join(scale_path, AMP_FILE)
            with open(amp_path, "w") as f:
                # Use float() to ensure compatibility if it's a tensor
                f.write(str(float(self.noise_amps[scale])))

    def save_scale(self, scale: int, path: str) -> None:
        """Save generator/discriminator and auxiliary files for a scale.

        The base implementation delegates framework-specific model saves to
        the concrete subclass hooks so the base remains agnostic about file
        formats and serialization APIs.

        Parameters
        ----------
        scale : int
            Pyramid scale index being saved.
        path : str
            Directory path where scale data should be saved.
        """
        # Ensure directory exists
        os.makedirs(path, exist_ok=True)

        # Framework-specific model state
        self.save_generator_state(path, scale)
        self.save_discriminator_state(path, scale)

        # Save amplitude and shape via subclass hooks (formats chosen by subclass)
        self.save_amp(path, scale)
        self.save_shape(path, scale)

    def setup_framework(self) -> None:
        """Create framework-specific objects and assign them to the instance.

        This generic helper calls the concrete `build_generator` and
        `create_discriminators_container` hooks and then moves the created
        generator to `device` using `move_to_device`. Subclasses may
        optionally override `move_to_device` to support framework-specific
        device placement.
        """
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()

    @abstractmethod
    def generate_noise(
        self,
        scale: int,
        indexes: torch.Tensor,
        well: Any | None = None,
        seismic: Any | None = None,
    ) -> Any:
        """Generate a noise tensor of given shape and batch size.

        Parameters
        ----------
        scale : int
            Pyramid scale index used to select the noise shape.
        indexes : tuple[int, ...]
            Tuple of batch/sample indices used to generate noise.
        well : Any | None, optional
            Well-conditioning tensor for the current scale.
        seismic : Any | None, optional
            Seismic-conditioning tensor for the current scale.

        Returns
        -------
        Any
            Generated noise tensor of shape (num_samp, *shape).

        Raises
        ------
        NotImplementedError
            If the subclass does not override this method.
        """
        raise NotImplementedError("Subclasses must implement generate_noise")
