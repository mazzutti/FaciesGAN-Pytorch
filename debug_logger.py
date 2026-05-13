"""Structured debug logger for FaciesGAN training pipeline diagnostics.

Writes JSONL entries (one JSON object per line) to::

    <output_path>/debug_train.jsonl

**Only rank-0 should call** :meth:`DebugLogger.initialize`.  All other
ranks call :meth:`DebugLogger.get` and receive a no-op ``_NoOpLogger``.

The logger throttles writes via a per-key call counter so that
high-frequency training loops do not become I/O bound:

    log_interval=10  →  one entry written every 10 calls per key.

Set the environment variable ``FACIESGAN_DEBUG_LOG=0`` to disable the
logger at runtime without code changes.

Usage
-----
::

    # One-time setup (trainer __init__, rank 0 only):
    from debug_logger import DebugLogger
    DebugLogger.initialize(output_path="outputs/run/debug", log_interval=10)

    # Log from any module:
    dl = DebugLogger.get()
    dl.log_tensor_stats("gen_output", "facies_logits_s0", tensor, step=epoch)
    dl.log_scalar("noise_amp_init", "rmse_s0", float(rmse), step=epoch)
    dl.log_dict("disc_scores/s0", {"d_real": -0.5, "d_fake": 0.3}, step=epoch)

JSONL schema
------------
Each line is a JSON object with at least the fields:

``event``
    One of ``"session_start"``, ``"tensor_stats"``, ``"scalar"``,
    ``"dict"``.
``tag``
    Logical group (e.g. ``"gen_output"``, ``"disc_scores/s0"``).
``name``
    Sub-key within the tag (e.g. ``"facies_logits_s0"``).
``step``
    Training epoch / iteration counter at the time of the call.
``ts``
    Unix timestamp (float seconds).

For ``"tensor_stats"`` entries an extra ``stats`` dict is included with
``mean``, ``std``, ``min``, ``max``, ``nan`` (count), ``inf`` (count).
For ``"scalar"`` entries a ``value`` float is included.
For ``"dict"`` entries a ``data`` dict of float values is included.
"""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Any

import torch


class DebugLogger:
    """Thread-safe JSONL debug logger with per-key throttling.

    Only one global instance is active at a time (singleton).  All
    methods on this class are safe to call from multiple threads; only
    one process/rank should call :meth:`initialize`.
    """

    _instance: "DebugLogger | None" = None
    _class_lock: threading.Lock = threading.Lock()

    def __init__(
        self,
        output_path: str,
        log_interval: int = 10,
        enabled: bool = True,
    ) -> None:
        env_flag = os.environ.get("FACIESGAN_DEBUG_LOG", "1")
        self._enabled: bool = enabled and env_flag != "0"
        self._log_interval: int = max(1, log_interval)
        self._counters: dict[str, int] = {}
        self._write_lock: threading.Lock = threading.Lock()
        self._fh: Any = None

        if self._enabled:
            os.makedirs(output_path, exist_ok=True)
            fpath = os.path.join(output_path, "debug_train.jsonl")
            self._fh = open(fpath, "a", buffering=1)  # line-buffered
            self._raw_write(
                {
                    "event": "session_start",
                    "ts": time.time(),
                    "pid": os.getpid(),
                    "log_interval": self._log_interval,
                }
            )

    # ── Class-level API ─────────────────────────────────────────────────────

    @classmethod
    def initialize(
        cls,
        output_path: str,
        log_interval: int = 10,
        enabled: bool = True,
    ) -> "DebugLogger":
        """Create (or replace) the process-global DebugLogger instance.

        Parameters
        ----------
        output_path:
            Directory where ``debug_train.jsonl`` will be written.
        log_interval:
            Write one entry every *log_interval* calls per (tag, name) key.
            Higher values reduce I/O overhead; lower values give finer
            resolution.  Default: 10.
        enabled:
            Set ``False`` to create a no-op instance (also honoured by the
            ``FACIESGAN_DEBUG_LOG=0`` environment variable).
        """
        with cls._class_lock:
            if cls._instance is not None:
                cls._instance.close()
            cls._instance = cls(output_path, log_interval, enabled)
        return cls._instance

    @classmethod
    def get(cls) -> "DebugLogger":
        """Return the active logger, or a no-op if :meth:`initialize` was
        not called (e.g. on non-rank-0 processes)."""
        inst = cls._instance
        return inst if inst is not None else _NoOpLogger()

    # ── Public logging API ──────────────────────────────────────────────────

    def log_tensor_stats(
        self,
        tag: str,
        name: str,
        tensor: torch.Tensor,
        step: int,
        expected_range: tuple[float, float] | None = None,
    ) -> None:
        """Log mean / std / min / max / nan-count / inf-count for *tensor*.

        Parameters
        ----------
        expected_range:
            Optional ``(lo, hi)`` tuple.  When provided the entry gains a
            ``range_ok`` bool and ``out_of_range_frac`` fraction (how many
            elements are outside ``[lo, hi]``), making violations immediately
            grep-able with ``"range_ok": false``.
        """
        key = f"{tag}/{name}"
        if not self._should_log(key):
            return
        with torch.no_grad():
            t = tensor.detach().float()
            stats: dict[str, float | int] = {
                "mean": float(t.mean().item()),
                "std": float(t.std().item()) if t.numel() > 1 else 0.0,
                "min": float(t.min().item()),
                "max": float(t.max().item()),
                "nan": int(t.isnan().sum().item()),
                "inf": int(t.isinf().sum().item()),
            }
            entry: dict[str, Any] = {
                "event": "tensor_stats",
                "tag": tag,
                "name": name,
                "step": step,
                "shape": list(tensor.shape),
                "stats": stats,
                "ts": time.time(),
            }
            if expected_range is not None:
                lo, hi = expected_range
                oor = int(((t < lo) | (t > hi)).sum().item())
                entry["expected_range"] = list(expected_range)
                entry["out_of_range"] = oor
                entry["range_ok"] = oor == 0
        self._raw_write(entry)

    def log_tensor_channel_stats(
        self,
        tag: str,
        name: str,
        tensor: torch.Tensor,
        step: int,
        channel_names: list[str] | None = None,
        expected_range: tuple[float, float] | None = None,
    ) -> None:
        """Log per-channel statistics for a (B, C, H, W) tensor.

        Produces one JSONL entry per channel so downstream analysis can
        pinpoint which channel is misbehaving without reading the whole
        batch at once.

        Parameters
        ----------
        channel_names:
            Optional list of length C for human-readable channel labels.
            Falls back to ``"ch0"``, ``"ch1"``, … if omitted.
        expected_range:
            Applied per-channel; ``range_ok`` is ``False`` if any element in
            that channel lies outside ``[lo, hi]``.
        """
        key = f"{tag}/{name}"
        if not self._should_log(key):
            return
        if tensor.ndim < 2:
            return
        C = tensor.shape[1] if tensor.ndim >= 2 else tensor.shape[0]
        with torch.no_grad():
            t = tensor.detach().float()
            channels: list[dict[str, Any]] = []
            for c in range(C):
                ch_name = (
                    channel_names[c]
                    if channel_names and c < len(channel_names)
                    else f"ch{c}"
                )
                if tensor.ndim == 4:
                    tc = t[:, c, ...]
                else:
                    tc = t[c, ...]
                ch_stats: dict[str, Any] = {
                    "ch": c,
                    "name": ch_name,
                    "mean": float(tc.mean().item()),
                    "std": float(tc.std().item()) if tc.numel() > 1 else 0.0,
                    "min": float(tc.min().item()),
                    "max": float(tc.max().item()),
                    "nan": int(tc.isnan().sum().item()),
                    "inf": int(tc.isinf().sum().item()),
                }
                if expected_range is not None:
                    lo, hi = expected_range
                    oor = int(((tc < lo) | (tc > hi)).sum().item())
                    ch_stats["expected_range"] = list(expected_range)
                    ch_stats["out_of_range"] = oor
                    ch_stats["range_ok"] = oor == 0
                channels.append(ch_stats)
        self._raw_write(
            {
                "event": "tensor_channel_stats",
                "tag": tag,
                "name": name,
                "step": step,
                "shape": list(tensor.shape),
                "channels": channels,
                "ts": time.time(),
            }
        )

    def log_facies_input(
        self,
        tag: str,
        tensor: torch.Tensor,
        step: int,
        num_facies_channels: int | None = None,
    ) -> None:
        """Domain-specific logger for real facies inputs (one-hot, B×C×H×W).

        Checks:
        - Values in {0, 1} (one-hot — range_ok with expected [0, 1])
        - Per-channel mean ≈ class frequency (should be >0 for all classes if
          the batch contains every facies)
        - Class distribution: fraction of pixels belonging to each class
        - Channel argmax is well-defined (no ties, every pixel has exactly one
          channel = 1)
        """
        key = f"{tag}/facies_input"
        if not self._should_log(key):
            return
        source_channels = tensor.shape[1] if tensor.ndim == 4 else tensor.shape[0]
        with torch.no_grad():
            t = tensor.detach().float()
            if num_facies_channels is not None:
                if t.ndim == 4 and t.shape[1] > num_facies_channels:
                    t = t[:, :num_facies_channels, ...]
                elif t.ndim == 3 and t.shape[0] > num_facies_channels:
                    t = t[:num_facies_channels, ...]
            channel_count = t.shape[1] if t.ndim == 4 else t.shape[0]
            ch_names = [f"facies_{i}" for i in range(channel_count)]
            # Per-channel frequency (fraction of pixels assigned to each class)
            class_freq = {
                ch_names[c]: float(t[:, c].mean().item()) for c in range(channel_count)
            }
            # Channel sums per pixel should equal 1 for perfect one-hot
            ch_sum = t.sum(dim=1)  # (B, H, W)
            sum_mean = float(ch_sum.mean().item())
            sum_max = float(ch_sum.max().item())
            sum_min = float(ch_sum.min().item())
            # How many pixels are not exactly one-hot?
            not_onehot = int(((ch_sum - 1.0).abs() > 1e-3).sum().item())
            # Fraction of values outside [0,1]
            oor = int(((t < 0.0) | (t > 1.0)).sum().item())
        self._raw_write(
            {
                "event": "facies_input_check",
                "tag": tag,
                "step": step,
                "shape": list(tensor.shape),
                "facies_shape": list(t.shape),
                "source_channels": source_channels,
                "range_ok": oor == 0,
                "out_of_range": oor,
                "onehot_ok": not_onehot == 0,
                "not_onehot_pixels": not_onehot,
                "ch_sum_mean": sum_mean,
                "ch_sum_min": sum_min,
                "ch_sum_max": sum_max,
                "class_freq": class_freq,
                "ts": time.time(),
            }
        )

    def log_rp_input(
        self,
        tag: str,
        tensor: torch.Tensor,
        step: int,
        channel_names: list[str] | None = None,
        normalization_range: tuple[float, float] = (0.0, 1.0),
    ) -> None:
        """Domain-specific logger for rock-physics tensors.

        Checks ``normalization_range`` per channel and flags out-of-range
        elements.
        Typical channel order: [Ip, Is, Vp/Vs].
        """
        lo = float(min(normalization_range))
        hi = float(max(normalization_range))
        ch_names = channel_names or ["Ip", "Is", "VpVs"]
        self.log_tensor_channel_stats(
            tag,
            "rp_channels",
            tensor,
            step,
            channel_names=ch_names,
            expected_range=(lo, hi),
        )

    def log_seismic_input(
        self,
        tag: str,
        tensor: torch.Tensor,
        step: int,
        normalization_range: tuple[float, float] = (0.0, 1.0),
    ) -> None:
        """Domain-specific logger for normalized seismic tensors.

        Checks ``normalization_range`` and logs mean.
        """
        lo = float(min(normalization_range))
        hi = float(max(normalization_range))
        self.log_tensor_stats(tag, "seismic", tensor, step, expected_range=(lo, hi))

    def log_wells_input(
        self,
        tag: str,
        tensor: torch.Tensor,
        step: int,
        num_facies_channels: int | None = None,
    ) -> None:
        """Domain-specific logger for wells tensors (one-hot, [0,1]).

        Channel 0 = background (no well); channels 1..num_facies_channels = facies at
        well location.  Checks range [0, 1] and that the well mask is sparse
        (most pixels are background).
        """
        C = tensor.shape[1] if tensor.ndim == 4 else tensor.shape[0]
        ch_names = ["bg"] + [f"facies_{i}" for i in range(1, C)]
        self.log_tensor_channel_stats(
            tag,
            "wells",
            tensor,
            step,
            channel_names=ch_names,
            expected_range=(0.0, 1.0),
        )

    def log_facies_output(
        self,
        tag: str,
        tensor: torch.Tensor,
        step: int,
        num_facies_channels: int,
    ) -> None:
        """Domain-specific logger for generator facies output (raw logits).

        Logs per-channel logit statistics and the entropy of the implied
        softmax distribution (high entropy → uniform / mode collapse;
        low entropy → confident predictions).
        """
        key = f"{tag}/facies_output"
        if not self._should_log(key):
            return
        ch_names = [f"logit_{i}" for i in range(num_facies_channels)]
        with torch.no_grad():
            t = tensor.detach().float()[:, :num_facies_channels]
            probs = torch.softmax(t, dim=1)
            # Shannon entropy per pixel: -sum(p * log(p)); mean over batch
            eps = 1e-8
            entropy = -(probs * (probs + eps).log()).sum(dim=1).mean().item()
            # Fraction of pixels where any single class dominates (prob > 0.8)
            confident_frac = float(
                (probs.max(dim=1).values > 0.8).float().mean().item()
            )
            # Class imbalance: how spread are the argmax predictions?
            argmax_cls = probs.argmax(dim=1)  # (B, H, W)
            class_counts = {
                f"argmax_cls_{c}": int((argmax_cls == c).sum().item())
                for c in range(num_facies_channels)
            }
        self._raw_write(
            {
                "event": "facies_output_check",
                "tag": tag,
                "step": step,
                "shape": list(tensor.shape),
                "softmax_entropy": float(entropy),
                "confident_frac": confident_frac,
                "argmax_class_counts": class_counts,
                "ts": time.time(),
            }
        )
        # Also log per-logit stats for saturation detection
        self.log_tensor_channel_stats(
            tag, "facies_logits", t, step, channel_names=ch_names
        )

    def log_rp_output(
        self,
        tag: str,
        tensor: torch.Tensor,
        step: int,
        channel_names: list[str] | None = None,
        normalization_range: tuple[float, float] = (0.0, 1.0),
    ) -> None:
        """Domain-specific logger for generator RP output.

        Flags values outside ``normalization_range`` (residual clamp failure).
        """
        lo = float(min(normalization_range))
        hi = float(max(normalization_range))
        ch_names = channel_names or ["Ip", "Is", "VpVs"]
        self.log_tensor_channel_stats(
            tag,
            "rp_output",
            tensor,
            step,
            channel_names=ch_names,
            expected_range=(lo, hi),
        )

    def log_scalar(self, tag: str, name: str, value: float, step: int) -> None:
        """Log a single named scalar value."""
        key = f"{tag}/{name}"
        if not self._should_log(key):
            return
        self._raw_write(
            {
                "event": "scalar",
                "tag": tag,
                "name": name,
                "value": float(value),
                "step": step,
                "ts": time.time(),
            }
        )

    def log_dict(self, tag: str, data: dict[str, float | int], step: int) -> None:
        """Log a flat dict of named scalar values under a common tag."""
        if not self._should_log(tag):
            return
        self._raw_write(
            {
                "event": "dict",
                "tag": tag,
                "data": {k: float(v) for k, v in data.items()},
                "step": step,
                "ts": time.time(),
            }
        )

    def close(self) -> None:
        """Flush and close the underlying log file."""
        if self._fh is not None:
            try:
                self._fh.flush()
                self._fh.close()
            except OSError:
                pass
            self._fh = None

    def __del__(self) -> None:
        self.close()

    # ── Internal helpers ────────────────────────────────────────────────────

    def _should_log(self, key: str) -> bool:
        if not self._enabled:
            return False
        count = self._counters.get(key, 0) + 1
        self._counters[key] = count
        return count % self._log_interval == 0

    def _raw_write(self, entry: dict[str, Any]) -> None:
        if self._fh is None:
            return
        line = json.dumps(entry, default=str) + "\n"
        with self._write_lock:
            self._fh.write(line)


class _NoOpLogger(DebugLogger):
    """Null-object returned on non-rank-0 processes or before initialization.

    All methods are no-ops so that call sites need no ``if logger is not None``
    guards.
    """

    def __init__(self) -> None:  # do NOT call super().__init__ — no file creation
        self._enabled = False

    def log_tensor_stats(self, *_: Any, **__: Any) -> None:
        pass

    def log_tensor_channel_stats(self, *_: Any, **__: Any) -> None:
        pass

    def log_facies_input(self, *_: Any, **__: Any) -> None:
        pass

    def log_rp_input(self, *_: Any, **__: Any) -> None:
        pass

    def log_seismic_input(self, *_: Any, **__: Any) -> None:
        pass

    def log_wells_input(self, *_: Any, **__: Any) -> None:
        pass

    def log_facies_output(self, *_: Any, **__: Any) -> None:
        pass

    def log_rp_output(self, *_: Any, **__: Any) -> None:
        pass

    def log_scalar(self, *_: Any, **__: Any) -> None:
        pass

    def log_dict(self, *_: Any, **__: Any) -> None:
        pass

    def close(self) -> None:
        pass
