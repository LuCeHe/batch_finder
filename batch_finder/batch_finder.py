"""
Batch Finder - Core functionality for finding maximum batch sizes along variable axes.
"""

import gc
import inspect
import json
import logging
import numbers
import os
import queue
import signal
import sys
import time
import warnings
import multiprocessing
import threading
from collections.abc import Mapping, Sequence
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
from tqdm import tqdm

from .input_shapes import (
    CONSTRAINTS_KEY,
    InputShapesSpec,
    materialize_compact_numeric_shape,
    materialize_shapes,
    normalize_compact_numeric_tuple,
    parse_input_shapes,
    parse_input_shapes_dict,
    collect_symbols_in_shapes,
)


def _materialize_int_shape_slots(shape: Tuple[int, ...], trial: int) -> Tuple[int, ...]:
    """
    Integer-only shape template: ``-1`` → ``trial``; ``d < -1`` → ``round(|d| * trial)`` (e.g. ``-2`` → ``2×`` trial);
    non-negative dims unchanged. Same scaling role as negative floats in compact mode.
    """
    out: List[int] = []
    for s in shape:
        if s == -1:
            out.append(trial)
        elif s < -1:
            out.append(int(round(abs(s) * trial)))
        else:
            out.append(s)
    return tuple(out)


def _parse_flat_shape_tuple(
    raw: Tuple[Any, ...],
) -> Tuple[str, Tuple[Any, ...]]:
    """
    One tensor: int-only ``single`` shape or ``compact`` numeric tuple.

    Int-only: at least one ``-1`` (searched axis). Other negative ints ``d < -1`` scale like
    compact floats: dimension ``round(|d| * trial)`` (e.g. ``-2`` → ``2×`` trial).

    Returns:
        ``("single", shape_ints)`` or ``("compact", spec)`` (same as ``normalize_compact_numeric_tuple`` output).
    """
    has_non_integral = False
    for x in raw:
        if isinstance(x, bool):
            raise TypeError("input_shapes: boolean is not a valid dimension")
        if isinstance(x, numbers.Integral):
            continue
        if isinstance(x, numbers.Real):
            has_non_integral = True
            break
        has_non_integral = True
        break
    if has_non_integral:
        spec = normalize_compact_numeric_tuple(raw)
        materialize_compact_numeric_shape(spec, 1)
        return "compact", spec
    try:
        shape = tuple(int(x) for x in raw)
    except (TypeError, ValueError) as e:
        raise TypeError(
            "When input_shapes is a tuple or list of integers, every element must be int-like "
            "(e.g. dimensions and -1 for the axis to maximize)."
        ) from e
    if -1 not in shape:
        raise ValueError(
            "tuple/list input_shapes must contain at least one -1 for the modifiable axis."
        )
    return "single", shape


def _normalize_input_shapes_arg(
    input_shapes: Optional[
        Union[str, Tuple[Any, ...], List[Any], Dict[str, Any]]
    ],
) -> Tuple[Optional[str], Any]:
    """
    Normalize ``input_shapes`` into a mode and payload.

    Returns:
        (None, None) — caller should use ``axis_to_maximize`` (no structured shapes).
        ("dsl", dsl_string) — parse with ``parse_input_shapes``.
        ("dict_dsl", dict) — parse with ``parse_input_shapes_dict`` after ``inputs_info`` is known.
        ("single", shape_tuple) — one tensor: tuple/list of ints with at least one ``-1``.
        ("compact", spec_tuple) — one tensor: ints + negative floats; searched axis is ``-1`` (int) or ``-1.0`` / ``-1.`` (float); other negative floats scale (e.g. ``-1.5`` → ``1.5×`` trial).
        ("multi_shape", list of (mode, payload)) — several tensors: same order as ``forward`` / ``forward_params``; each entry is ``("single", ...)`` or ``("compact", ...)`` from ``_parse_flat_shape_tuple``.
    """
    if input_shapes is None:
        return None, None
    if isinstance(input_shapes, str):
        s = input_shapes.strip()
        if not s:
            raise ValueError("input_shapes string is empty.")
        return "dsl", s
    if isinstance(input_shapes, Mapping):
        return "dict_dsl", dict(input_shapes)
    if isinstance(input_shapes, (tuple, list)):
        if len(input_shapes) == 0:
            raise ValueError("input_shapes tuple/list is empty.")
        # List/tuple of per-tensor shapes: [(-1, 128, 512), (-1, 128, 512)] — not a flat (-1, …).
        if isinstance(input_shapes[0], (tuple, list)):
            parts: List[Tuple[str, Tuple[Any, ...]]] = []
            for g in input_shapes:
                if not isinstance(g, (tuple, list)):
                    raise TypeError(
                        "input_shapes: with multiple tensors, each element must be a tuple or list "
                        "of dimensions, e.g. [(-1, 128, 512), (-1, 128, 512)]."
                    )
                parts.append(_parse_flat_shape_tuple(tuple(g)))
            return "multi_shape", parts
        return _parse_flat_shape_tuple(tuple(input_shapes))
    raise TypeError(
        f"input_shapes must be str (DSL), dict, or tuple/list of int/float; got {type(input_shapes)}"
    )


# Typical HuggingFace causal LM tensor args (``forward``), in signature order for GPT2-style models.
# Pass ``forward_params=...`` to override when your checkpoint differs.
DEFAULT_FORWARD_PARAMS_CAUSAL_LM: Tuple[str, ...] = (
    "input_ids",
    "attention_mask",
    "token_type_ids",
    "position_ids",
    "labels",
)


def _infer_input_type(param_name: str, dtype_overrides: Optional[Dict[str, str]] = None) -> str:
    """Infer whether an input expects integer or float tensors."""
    if dtype_overrides and param_name in dtype_overrides:
        return dtype_overrides[param_name]
    name_lower = param_name.lower()
    if any(kw in name_lower for kw in ("ids", "mask", "labels", "indices", "token_type")):
        return "integer"
    # Default for hidden_states, pixel_values, embeddings, x, input, etc.
    return "float"


def _inputs_info_from_names(names: List[str]) -> List[Tuple[str, inspect.Parameter]]:
    """Synthetic forward parameters when names come from ``input_shapes`` dict or ``forward_params``."""
    return [
        (n, inspect.Parameter(n, inspect.Parameter.POSITIONAL_OR_KEYWORD))
        for n in names
    ]


# HuggingFace-style ``forward`` kwargs that are not probed as separate tensor slots.
# Optional encoder / cross-attn args (e.g. newer GPT2 ``ForCausalLM``) must not get dummy
# tensors during probing—HF raises if they are set inconsistently. Use ``forward_params=``
# for encoder–decoder models where these are real inputs.
_SKIP_FORWARD_PARAMS = frozenset(
    {
        "return_dict",
        "output_attentions",
        "output_hidden_states",
        "output_router_logits",
        "output_scores",
        "output_logits",
        "use_cache",
        "past_key_values",
        "cache_position",
        "past_bucket_indices",
        "head_mask",
        "cross_attn_head_mask",
        "decoder_head_mask",
        # Alternate to ``input_ids``; passing both often breaks HF models.
        "inputs_embeds",
        "decoder_inputs_embeds",
        "encoder_hidden_states",
        "encoder_attention_mask",
        "logits_to_keep",
    }
)


def _detect_model_inputs(
    model: torch.nn.Module,
) -> List[Tuple[str, inspect.Parameter]]:
    """Tensor-like ``forward`` parameters in signature order (after ``_SKIP_FORWARD_PARAMS``)."""
    sig = inspect.signature(model.forward)
    out: List[Tuple[str, inspect.Parameter]] = []
    for name, p in sig.parameters.items():
        if name == "self":
            continue
        if p.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        if name in _SKIP_FORWARD_PARAMS:
            continue
        out.append((name, p))
    return out


def _subprocess_probe_worker(
    result_queue: Any,
    get_model: Callable[[], torch.nn.Module],
    cuda_devs: List[int],
    device_str: str,
) -> None:
    """Child entry: one ``get_model()`` call; put ``("ok", config, names)`` or ``("err", msg)``."""
    try:
        device = torch.device(device_str)
        m = get_model()
        try:
            cfg = getattr(m, "config", None)
            names = [n for n, _ in _detect_model_inputs(m)]
            result_queue.put(("ok", cfg, names))
        finally:
            del m
            gc.collect()
            _release_memory(device, cuda_devs, aggressive=True)
    except Exception as e:
        try:
            result_queue.put(("err", repr(e)))
        except Exception:
            pass


def _get_hidden_size_from_config(config: Optional[Any]) -> int:
    """Infer hidden/embedding size from a HuggingFace-style ``config`` object."""
    if config is not None:
        for attr in ("hidden_size", "embed_dim", "d_model"):
            v = getattr(config, attr, None)
            if v is not None:
                return v
    return 768


def _estimate_shape_from_config(
    param_name: str, config: Optional[Any]
) -> Optional[Tuple[int, ...]]:
    """Rough shape hints from ``config`` only (no live module in the parent)."""
    if config is None:
        return None
    name_lower = param_name.lower()
    if "input_ids" in name_lower or "attention_mask" in name_lower or "token_type" in name_lower:
        seq = getattr(config, "max_position_embeddings", None) or getattr(config, "max_seq_length", None) or 512
        return (2, seq)
    if "pixel" in name_lower:
        h = getattr(config, "image_size", 224)
        if not isinstance(h, int) and hasattr(config, "vision_config"):
            vc = getattr(config, "vision_config", None)
            h = getattr(vc, "image_size", 224) if vc else 224
        h = h if isinstance(h, int) else 224
        return (2, 3, h, h)
    return None


def _get_default_shape_for_param(
    param_name: str,
    axis_values: Dict[str, int],
    config: Optional[Any],
) -> Tuple[int, ...]:
    """Default shape from naming conventions + optional ``config`` (HF-style)."""
    name_lower = param_name.lower()

    batch = axis_values.get("batch_size", 2)
    seq_len = axis_values.get("seq_len", 64)
    n_docs = axis_values.get("n_docs", 1)
    hidden = _get_hidden_size_from_config(config)

    if "encoder" in name_lower and "input" in name_lower:
        return (n_docs, batch, seq_len)
    if "encoder" in name_lower and "mask" in name_lower:
        return (n_docs, batch, seq_len)

    if "input_ids" in name_lower or "attention_mask" in name_lower or "token_type" in name_lower:
        return (batch, seq_len)
    if "position_ids" in name_lower:
        return (batch, seq_len)
    if "labels" in name_lower or "label" in name_lower:
        return (batch, seq_len)

    if "pixel" in name_lower:
        return (batch, 3, 224, 224)

    if name_lower in ("x", "input", "hidden_states", "inputs_embeddings"):
        est = _estimate_shape_from_config(param_name, config)
        if est is not None and len(est) >= 3:
            return (batch, seq_len, est[2])
        return (batch, seq_len, hidden)

    est = _estimate_shape_from_config(param_name, config)
    if est is not None:
        out = list(est)
        if len(out) >= 1:
            out[0] = batch
        if len(out) >= 2:
            out[1] = seq_len
        return tuple(out)

    return (batch, seq_len)


def _describe_subprocess_exitcode(code: Optional[int]) -> str:
    """Human-readable child exit code (negative values are ``-signal`` on POSIX)."""
    if code is None:
        return "?"
    if code >= 0:
        return str(code)
    sig = -code
    try:
        return f"{code} ({signal.Signals(sig).name})"
    except (ValueError, AttributeError):
        return str(code)


def _default_use_subprocess(_device: torch.device) -> bool:
    """
    Linux/macOS: default **subprocess** for CUDA, CPU, and MPS.

    Each attempt runs in a **child** so host OOM (SIGKILL) or CUDA OOM handling does not
    terminate the parent interpreter—only the worker dies and the search continues.

    Set ``BATCH_FINDER_SUBPROCESS=0`` for **in-process** attempts (e.g. shared login nodes
    where repeated full HF reloads per spawn are worse than one process).

    Windows: always in-process (no attempt subprocess). Env is ignored except for docs.
    """
    if sys.platform == "win32":
        return False
    env = os.environ.get("BATCH_FINDER_SUBPROCESS", "").strip().lower()
    if env in ("0", "false", "no", "off"):
        return False
    if env in ("1", "true", "yes", "on"):
        return True
    return True


def _process_rss_bytes() -> int:
    """Best-effort resident set size (bytes). Linux ru_maxrss is KiB; macOS is bytes."""
    try:
        import resource as _res

        rss = int(_res.getrusage(_res.RUSAGE_SELF).ru_maxrss)
        if sys.platform == "darwin":
            return rss
        return rss * 1024
    except Exception:
        return 0


def _resolve_cuda_mem_devices(
    device: torch.device,
    cuda_mem_devices: Optional[Union[int, Sequence[int], str]],
) -> List[int]:
    """
    Which CUDA device indices to reset/read for peak memory stats.
    ``None`` → only ``device``'s index. ``\"all\"`` → every visible GPU.
    """
    if device.type != "cuda" or not torch.cuda.is_available():
        return []
    nd = torch.cuda.device_count()
    if cuda_mem_devices is None:
        idx = device.index if device.index is not None else torch.cuda.current_device()
        return [int(idx) if idx >= 0 else 0]
    if isinstance(cuda_mem_devices, str):
        if cuda_mem_devices.strip().lower() == "all":
            return list(range(nd))
        raise ValueError('cuda_mem_devices as str must be "all"')
    if isinstance(cuda_mem_devices, int):
        devs = [int(cuda_mem_devices)]
    else:
        devs = [int(x) for x in cuda_mem_devices]
    out = sorted(set(devs))
    for d in out:
        if d < 0 or d >= nd:
            raise ValueError(f"cuda_mem_devices: invalid GPU index {d} (device_count={nd})")
    return out


def _cuda_multiplier_from_peaks(
    peak_by_dev: Dict[int, int],
    target_fraction: float,
    max_multiplier: float,
) -> Optional[float]:
    """
    Per-GPU linear headroom; return the **minimum** multiplier so the tightest GPU limits growth.
    """
    ms: List[float] = []
    for dev, peak in peak_by_dev.items():
        if peak <= 0:
            continue
        try:
            tot = int(torch.cuda.get_device_properties(dev).total_memory)
            m = (target_fraction * float(tot)) / float(peak)
            ms.append(min(max_multiplier, max(1.05, m)))
        except Exception:
            pass
    return min(ms) if ms else None


def _memory_guided_success_multiplier(
    peak_by_dev: Dict[int, int],
    rss_delta: int,
    *,
    target_fraction: float,
    max_multiplier: float,
    fallback: float,
) -> float:
    """
    How much to multiply the current batch after a successful step.
    Multi-GPU: bottleneck is the GPU with least headroom (min multiplier).
    """
    candidates: List[float] = []
    m = _cuda_multiplier_from_peaks(peak_by_dev, target_fraction, max_multiplier)
    if m is not None:
        candidates.append(m)
    if rss_delta > 0:
        try:
            import psutil

            avail = float(psutil.virtual_memory().available)
            m2 = min(max_multiplier, max(1.05, 1.0 + 0.45 * (avail / float(rss_delta))))
            candidates.append(m2)
        except Exception:
            pass
    if not candidates:
        return min(max_multiplier, max(1.05, fallback))
    return min(min(candidates), max_multiplier)


def _release_memory(device: torch.device, cuda_devs: List[int], aggressive: bool = False) -> None:
    """Free allocator memory after a failed attempt (OOM paths)."""
    if aggressive:
        gc.collect()
    if device.type == "cuda" and torch.cuda.is_available() and cuda_devs:
        for d in cuda_devs:
            try:
                with torch.cuda.device(d):
                    torch.cuda.empty_cache()
            except Exception:
                pass
    elif device.type == "mps":
        try:
            if hasattr(torch, "mps") and torch.mps.is_available():
                torch.mps.empty_cache()
        except Exception:
            pass


def _default_delay_for_device(device: torch.device, delay: Optional[float]) -> float:
    if delay is not None:
        return delay
    return 3.0 if device.type == "cuda" else 0.75


def _default_initial_value_for_device(device: torch.device, initial_value: Optional[int]) -> int:
    if initial_value is not None:
        return initial_value
    if device.type == "cuda":
        return 512
    if device.type == "mps":
        return 64
    return 32


def _default_distributed_sync_dir() -> str:
    """Shared-ish cache dir for multi-rank JSON sync (HPC ``WORK``, else home)."""
    work = os.environ.get("WORK", "").strip()
    if work:
        return os.path.join(work, ".cache")
    return os.path.join(os.path.expanduser("~"), ".cache")


def _distributed_sync_job_tag() -> str:
    """Subdirectory name shared by all ranks in one launcher job (isolates stale sync JSON)."""
    for key in ("SLURM_JOB_ID", "LSB_JOBID", "PBS_JOBID", "TORCHELASTIC_RUN_ID"):
        v = os.environ.get(key, "").strip()
        if v:
            return v
    ma = os.environ.get("MASTER_ADDR", "").strip()
    mp = os.environ.get("MASTER_PORT", "").strip()
    if ma and mp:
        return f"{ma}_{mp}".replace(os.sep, "_").replace(":", "_")
    return "default"


def find_max_minibatch(
    get_model: Callable[[], torch.nn.Module],
    axis_to_maximize: Optional[str] = None,
    fixed_axis: Optional[Dict[str, int]] = None,
    device: Optional[torch.device] = None,
    delay: Optional[float] = None,
    initial_value: Optional[int] = None,
    n_attempts: int = 50,
    inference_only: bool = False,
    factor_down: float = 2.0,
    factor_up: float = 2.0,
    memory_guided: bool = True,
    memory_target_fraction: float = 0.88,
    max_growth_multiplier: float = 6.0,
    cuda_mem_devices: Optional[Union[int, Sequence[int], str]] = None,
    input_shapes: Optional[
        Union[str, Tuple[Any, ...], List[Any], Dict[str, Any]]
    ] = None,
    use_subprocess: Optional[bool] = None,
    *,
    forward_params: Optional[Sequence[str]] = None,
    time_limit_seconds: Optional[float] = None,
    distributed_sync_dir: Optional[str] = None,
) -> Optional[Union[int, Tuple[int, ...], Tuple[Tuple[int, ...], ...]]]:
    """
    Find the maximum value for the modifiable axis that the model can process without OOM.

    Supports several modes (via ``input_shapes`` or ``axis_to_maximize``):

    1. **String (DSL):** one ``(...)`` group per forward tensor argument, shared symbols,
       optional constraints (e.g. ``t=1.5b``), and exactly one ``symbol=-1`` for the
       dimension to maximize.
    2. **Dict (named DSL):** keys are ``forward`` parameter names; values are shape strings
       like ``\"(b, t), int\"`` or ``\"(d, b, t)\"``; include ``\"#constraints\"`` for
       constraints (e.g. ``\"t=2b, b=-1\"``). Clearer for models with many arguments.
    3. **Tuple or list (single tensor):** only the first ``forward`` argument; must include
       at least one ``-1`` (all ints), **or** a **compact numeric** tuple mixing ``int`` and
       negative ``float`` factors (e.g. ``(-1, 4, -1.5, 16)`` or ``(-1., 4, -1.3, 16)``): exactly one searched axis as int ``-1`` or float ``-1.``; other negative floats give size
       ``round(|f| * s)`` where ``s`` is the trial size on the searched axis.
       Int-only shapes may also use integers ``d < -1`` (e.g. ``-2``) for ``round(|d| * s)``, same as negative floats.
    3b. **List/tuple of shape tuples (several tensors):** e.g.
       ``[(-1, 128, 512), (-1, 128, 512)]`` — one shape per ``forward`` tensor, same order as
       ``forward_params`` / signature; each entry uses the same rules as (3). The searched
       trial size is shared across every ``-1`` (and compact scaled dims) in all tensors.
    4. **axis_to_maximize + fixed_axis:** when ``input_shapes`` is omitted; for multi-input
       models (e.g. HuggingFace) by symbolic axis name.

    Search strategy: by default **memory-guided** on success (GPU peak VRAM vs device total,
    optional CPU via RSS + psutil), capped by ``max_growth_multiplier`` (e.g. jump toward
    ~6× in one step when headroom allows). On failure outside a bracket, divide by
    ``factor_down``. When a success and a failure bracket the optimum, the next trial is always
    the integer midpoint ``(max_ok + min_fail) // 2`` (binary search). Set ``memory_guided=False``
    to use only ``factor_up`` / ``factor_down`` after successes.

    Args:
        get_model: Callable returning a fresh ``nn.Module`` per attempt (or per subprocess).
            Picklable at top level under ``spawn`` (prefer a module-level function, not a
            ``lambda``). Before the search loop, batch_finder loads the model **once** to read
            ``.config`` (if present) and infer ordered ``forward`` tensor argument names (same
            skips as HuggingFace-style flags). With default subprocess mode (Linux/macOS),
            that probe runs in a **child** so the parent never holds weights. Each attempt still
            builds a new module; workers **delete** the model and clear CUDA/MPS caches when the
            attempt finishes.
        axis_to_maximize: Name of axis to maximize when ``input_shapes`` is None.
        fixed_axis: Dict of fixed symbol or axis values (DSL extra symbols or HF-style keys).
        device: Device to run on (default: cuda if available else cpu).
        delay: Seconds between attempts; default ``None`` → 3.0 on CUDA, 0.75 on CPU/MPS.
        initial_value: First size to try; default ``None`` → 512 on CUDA, 64 on MPS, 32 on CPU.
        n_attempts: Maximum attempts.
        inference_only: If True, skip forward gradients and backward pass. If False, runs full forward+backward.
        factor_down: On failure, next = value / factor_down (default 2).
        factor_up: On success when ``memory_guided=False``, next ≈ value * factor_up.
        memory_guided: Use measured GPU/CPU memory to pick the next trial size (default True).
        memory_target_fraction: Aim peak VRAM at this fraction of total GPU memory (e.g. 0.88).
        max_growth_multiplier: Maximum single-step multiplicative increase after success (e.g. 6).
        cuda_mem_devices: CUDA indices for peak memory stats. ``None`` = ``device``'s index only;
            ``\"all\"`` or ``[0, 1, …]`` for multi-GPU bottleneck. Ignored when not on CUDA.
        input_shapes: DSL string, dict (named shapes + ``#constraints``), flat tuple/list of ints
            or compact numeric tuple, or a **list/tuple of per-tensor shapes**
            ``[(-1, 128, 512), …]`` (see docstring).
        use_subprocess: If ``True``, each attempt runs in a child (Linux/macOS). If ``False``,
            in-process. If ``None``, default is subprocess on **CUDA, CPU, and MPS** (worker
            may be OOM-killed without killing the parent). Set ``BATCH_FINDER_SUBPROCESS=0``
            for in-process on memory-tight hosts. Windows: in-process.
        forward_params: Optional override for ordered ``forward`` tensor argument names when
            ``input_shapes`` is not a dict. If omitted, names are taken from
            ``inspect.signature(model.forward)`` (skipping non-input kwargs such as ``use_cache``),
            then if that is empty and you use ``axis_to_maximize`` only, defaults to
            :data:`DEFAULT_FORWARD_PARAMS_CAUSAL_LM`. Override when inference misses an argument
            or order does not match your DSL/tuple.
        time_limit_seconds: If set, wall-clock limit **from the start of the search loop** (after
            the config/signature probe). When time is up, returns the **best batch found so far**
            (largest successful trial size) if any, else ``None``. Between attempts, sleeps are
            shortened so the limit is respected. With ``use_subprocess=True``, a running worker is
            **terminated** (``terminate`` / ``kill``) if it would exceed the remaining time.
            In-process attempts are not interrupted mid-forward; the limit applies between attempts.
        distributed_sync_dir: Used only when ``WORLD_SIZE > 1`` (e.g. ``torchrun``, Accelerate).
            Each rank may finish with a different best **trial size**; before returning, the
            function takes the **minimum** across ranks via JSON files so every process agrees
            on the same limit for DDP. The path must be **identical on every rank** (same shared
            tree). Do not pass a per-process random ``output_dir`` default. Prefer unset (then
            ``BATCH_FINDER_SYNC_DIR`` or ``$WORK/.cache`` / ``$HOME/.cache``). Files go under
            ``…/find_max_minibatch_sync/<job_tag>/`` (``SLURM_JOB_ID``, ``MASTER_ADDR``+``MASTER_PORT``, …).

    Returns:
        For int-only tuple/list ``input_shapes``: final shape tuple with each ``-1`` replaced
            by the max value.
        For compact numeric ``input_shapes``: final materialized shape tuple (ints only).
        For **multi_shape** (list of per-tensor shapes): tuple of final shape tuples, one per
            tensor (same order as ``forward``).
        For DSL string or ``axis_to_maximize``: int (max symbol or axis value).
        None if no value succeeded (including when the time limit expires before any success).
    """
    if time_limit_seconds is not None and float(time_limit_seconds) < 0:
        raise ValueError("time_limit_seconds must be non-negative when set.")
    fixed_axis = fixed_axis or {}
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    delay = _default_delay_for_device(device, delay)
    initial_value = _default_initial_value_for_device(device, initial_value)
    cuda_devs = _resolve_cuda_mem_devices(device, cuda_mem_devices)

    shape_mode, shape_payload = _normalize_input_shapes_arg(input_shapes)

    if shape_mode is not None and axis_to_maximize is not None:
        raise ValueError("Do not combine input_shapes with axis_to_maximize.")

    # Subprocess vs in-process (same rule as the search loop; needed before the probe load).
    mp_ctx = multiprocessing.get_context("spawn") if sys.platform != "win32" else None
    mp_ok = bool(
        mp_ctx is not None
        and (use_subprocess if use_subprocess is not None else _default_use_subprocess(device))
    )

    # One load: ``.config`` for vocab / hints + ``forward`` tensor arg names (signature, after skips).
    # When ``mp_ok``, the probe runs in a child so the parent never holds weights.
    cfg_effective: Optional[Any] = None
    probe_detected_names: List[str] = []
    if mp_ok and mp_ctx is not None:
        probe_q: Any = mp_ctx.Queue()
        probe_proc = mp_ctx.Process(
            target=_subprocess_probe_worker,
            args=(probe_q, get_model, cuda_devs, str(device)),
        )
        probe_proc.start()
        probe_proc.join()
        if probe_proc.exitcode != 0:
            raise RuntimeError(
                f"batch_finder probe subprocess exited with code {probe_proc.exitcode}"
            )
        try:
            probe_msg: Any = probe_q.get_nowait()
        except queue.Empty as e:
            raise RuntimeError("batch_finder probe subprocess returned no result") from e
        finally:
            try:
                probe_q.close()
                probe_q.join_thread()
            except Exception:
                pass
        if probe_msg[0] == "err":
            raise RuntimeError(f"batch_finder probe failed: {probe_msg[1]}")
        _, cfg_effective, probe_detected_names = probe_msg
    else:
        _probe = get_model()
        try:
            cfg_effective = getattr(_probe, "config", None)
            probe_detected_names = [n for n, _ in _detect_model_inputs(_probe)]
        finally:
            del _probe
            gc.collect()
            _release_memory(device, cuda_devs, aggressive=True)

    # Forward tensor arguments: dict keys, explicit forward_params, then signature probe, then HF default.
    inputs_info: List[Tuple[str, inspect.Parameter]]
    if shape_mode == "dict_dsl":
        assert shape_payload is not None
        ordered = [k for k in shape_payload if k != CONSTRAINTS_KEY]
        if not ordered:
            raise ValueError(
                f"input_shapes dict must include tensor keys besides {CONSTRAINTS_KEY!r}."
            )
        inputs_info = _inputs_info_from_names(ordered)
    elif forward_params is not None:
        inputs_info = _inputs_info_from_names(list(forward_params))
    elif probe_detected_names:
        inputs_info = _inputs_info_from_names(probe_detected_names)
    elif axis_to_maximize is not None:
        inputs_info = _inputs_info_from_names(list(DEFAULT_FORWARD_PARAMS_CAUSAL_LM))
    else:
        raise ValueError(
            "Pass input_shapes as a dict (named DSL), or forward_params=[...] with tensor "
            "argument names in order (same order as string DSL groups or tuple/compact modes), "
            "or use axis_to_maximize without input_shapes (default causal-LM names), or ensure "
            "get_model().forward exposes inferrable tensor parameters."
        )

    use_single_shape = shape_mode == "single"
    use_compact_shape = shape_mode == "compact"
    use_multi_shape = shape_mode == "multi_shape"
    single_shape: Optional[Tuple[int, ...]] = shape_payload if use_single_shape else None
    compact_spec: Optional[Tuple[Union[int, float], ...]] = (
        shape_payload if use_compact_shape else None
    )
    multi_shape_parts: Optional[List[Tuple[str, Any]]] = (
        shape_payload if use_multi_shape else None
    )
    spec: Optional[InputShapesSpec] = None
    dtype_overrides: Optional[Dict[str, str]] = None

    if use_multi_shape and multi_shape_parts is not None:
        if len(multi_shape_parts) != len(inputs_info):
            raise ValueError(
                f"input_shapes lists {len(multi_shape_parts)} tensor shape(s); "
                f"forward has {len(inputs_info)} tensor argument(s) after skips."
            )

    if shape_mode == "dsl":
        assert shape_payload is not None
        spec = parse_input_shapes(shape_payload)
    elif shape_mode == "dict_dsl":
        assert shape_payload is not None
        spec, dtype_overrides = parse_input_shapes_dict(
            shape_payload, [n for n, _ in inputs_info]
        )

    use_dsl = spec is not None

    if use_dsl:
        assert spec is not None
        if len(spec.shapes) != len(inputs_info):
            tqdm.write("\n--- batch_finder: forward vs input_shapes (mismatch) ---")
            tqdm.write(
                f"  input_shapes has {len(spec.shapes)} tuple(s); "
                f"forward() has {len(inputs_info)} tensor argument(s) after skips."
            )
            tqdm.write("  forward signature (order matters, same as input_shapes tuples):")
            try:
                for i, (name, p) in enumerate(inputs_info):
                    tqdm.write(f"    [{i}] {name}: {p}")
            except Exception as e:
                tqdm.write(f"    (could not list: {e})")
            tqdm.write("  Parsed input_shapes shapes:")
            for i, sh in enumerate(spec.shapes):
                tqdm.write(f"    [{i}] {tuple(sh)}")
            tqdm.write("  Full forward signature (including skipped params):")
            if probe_detected_names:
                tqdm.write(
                    f"    inferred tensor args (after batch_finder skips): {probe_detected_names}"
                )
            else:
                tqdm.write(
                    "    (none inferred — pass forward_params / dict keys to match forward)"
                )
            tqdm.write("---\n")
            raise ValueError(
                f"input_shapes has {len(spec.shapes)} tensor(s) but forward has {len(inputs_info)} tensor argument(s)."
            )
        syms = collect_symbols_in_shapes(spec)
        if spec.search_symbol not in syms:
            raise ValueError(
                f"search symbol {spec.search_symbol!r} must appear in at least one shape tuple."
            )

    if (
        not use_dsl
        and not use_single_shape
        and not use_compact_shape
        and not use_multi_shape
        and not axis_to_maximize
    ):
        raise ValueError("Must provide input_shapes or axis_to_maximize.")

    shape_param_name = (
        inputs_info[0][0]
        if (use_single_shape or use_compact_shape or use_multi_shape)
        else None
    )

    # Build sample axis_values for size estimation
    _sample_axis_values = {**fixed_axis}
    if use_dsl and spec is not None:
        _, binds = materialize_shapes(spec, initial_value, fixed_axis)
        _sample_axis_values.update(binds)
    elif use_single_shape and single_shape is not None:
        _sample_shape = _materialize_int_shape_slots(single_shape, initial_value)
        _sample_axis_values["batch_size"] = _sample_shape[0] if _sample_shape else initial_value
        _sample_axis_values["seq_len"] = _sample_shape[1] if len(_sample_shape) > 1 else 64
    elif use_compact_shape and compact_spec is not None:
        _sample_shape = materialize_compact_numeric_shape(compact_spec, initial_value)
        _sample_axis_values["batch_size"] = _sample_shape[0] if _sample_shape else initial_value
        _sample_axis_values["seq_len"] = _sample_shape[1] if len(_sample_shape) > 1 else 64
    elif use_multi_shape and multi_shape_parts is not None:
        m0, p0 = multi_shape_parts[0]
        if m0 == "single":
            _sample_shape = _materialize_int_shape_slots(p0, initial_value)
        else:
            _sample_shape = materialize_compact_numeric_shape(p0, initial_value)
        _sample_axis_values["batch_size"] = _sample_shape[0] if _sample_shape else initial_value
        _sample_axis_values["seq_len"] = _sample_shape[1] if len(_sample_shape) > 1 else 64
    else:
        _sample_axis_values[axis_to_maximize] = initial_value

    tqdm.write("\n\n--- Detected inputs (type, estimated shape) ---")
    for idx, (name, _) in enumerate(inputs_info):
        itype = _infer_input_type(name, dtype_overrides)
        if use_dsl and spec is not None:
            shapes_list, _ = materialize_shapes(spec, initial_value, fixed_axis)
            shape = shapes_list[idx]
        elif use_multi_shape and multi_shape_parts is not None and idx < len(multi_shape_parts):
            m_i, p_i = multi_shape_parts[idx]
            if m_i == "single":
                shape = _materialize_int_shape_slots(p_i, initial_value)
            else:
                shape = materialize_compact_numeric_shape(p_i, initial_value)
        elif use_compact_shape and idx == 0 and compact_spec is not None:
            shape = materialize_compact_numeric_shape(compact_spec, initial_value)
        elif use_single_shape and idx == 0 and single_shape is not None:
            shape = _materialize_int_shape_slots(single_shape, initial_value)
        else:
            shape = _get_default_shape_for_param(name, _sample_axis_values, cfg_effective)
        tqdm.write(f"  {name}: {itype}, {shape}")
    tqdm.write("---")

    # Float inputs must require grad when training-mode probing so loss.backward() has a graph.
    # (Models with no nn.Parameter still need leaf tensors with requires_grad=True.)
    _float_requires_grad = not inference_only

    def make_inputs(value: int) -> Dict[str, torch.Tensor]:
        inputs = {}
        if use_dsl and spec is not None:
            shapes_list, _ = materialize_shapes(spec, value, fixed_axis)
            for (param_name, _), shape in zip(inputs_info, shapes_list):
                dtype = _infer_input_type(param_name, dtype_overrides)
                if dtype == "integer":
                    vocab = getattr(cfg_effective, "vocab_size", None) or 50257
                    inputs[param_name] = torch.randint(0, min(vocab, 1000), shape, device=device, dtype=torch.long)
                else:
                    inputs[param_name] = torch.randn(
                        shape, device=device, requires_grad=_float_requires_grad
                    )
        elif use_multi_shape and multi_shape_parts is not None:
            for idx, (param_name, _) in enumerate(inputs_info):
                mode_i, payload_i = multi_shape_parts[idx]
                if mode_i == "single":
                    shape = _materialize_int_shape_slots(payload_i, value)
                else:
                    shape = materialize_compact_numeric_shape(payload_i, value)
                dtype = _infer_input_type(param_name, dtype_overrides)
                if dtype == "integer":
                    vocab = getattr(cfg_effective, "vocab_size", None) or 50257
                    inputs[param_name] = torch.randint(
                        0, min(vocab, 1000), shape, device=device, dtype=torch.long
                    )
                else:
                    inputs[param_name] = torch.randn(
                        shape, device=device, requires_grad=_float_requires_grad
                    )
        elif use_compact_shape and compact_spec is not None:
            shape = materialize_compact_numeric_shape(compact_spec, value)
            dtype = _infer_input_type(shape_param_name)
            if dtype == "integer":
                vocab = getattr(cfg_effective, "vocab_size", None) or 50257
                inputs[shape_param_name] = torch.randint(0, min(vocab, 1000), shape, device=device, dtype=torch.long)
            else:
                inputs[shape_param_name] = torch.randn(
                    shape, device=device, requires_grad=_float_requires_grad
                )
            for param_name, _ in inputs_info[1:]:
                axis_values = {"batch_size": shape[0] if len(shape) >= 1 else value, "seq_len": shape[1] if len(shape) >= 2 else 64}
                p_shape = _get_default_shape_for_param(param_name, axis_values, cfg_effective)
                p_dtype = _infer_input_type(param_name)
                if p_dtype == "integer":
                    vocab = getattr(cfg_effective, "vocab_size", None) or 50257
                    inputs[param_name] = torch.randint(0, min(vocab, 1000), p_shape, device=device, dtype=torch.long)
                else:
                    inputs[param_name] = torch.randn(
                        p_shape, device=device, requires_grad=_float_requires_grad
                    )
        elif use_single_shape and single_shape is not None:
            shape = _materialize_int_shape_slots(single_shape, value)
            dtype = _infer_input_type(shape_param_name)
            if dtype == "integer":
                vocab = getattr(cfg_effective, "vocab_size", None) or 50257
                inputs[shape_param_name] = torch.randint(0, min(vocab, 1000), shape, device=device, dtype=torch.long)
            else:
                inputs[shape_param_name] = torch.randn(
                    shape, device=device, requires_grad=_float_requires_grad
                )
            # For multi-input: broadcast compatible shapes to other params (e.g. attention_mask)
            for param_name, _ in inputs_info[1:]:
                axis_values = {"batch_size": shape[0] if len(shape) >= 1 else value, "seq_len": shape[1] if len(shape) >= 2 else 64}
                p_shape = _get_default_shape_for_param(param_name, axis_values, cfg_effective)
                p_dtype = _infer_input_type(param_name)
                if p_dtype == "integer":
                    vocab = getattr(cfg_effective, "vocab_size", None) or 50257
                    inputs[param_name] = torch.randint(0, min(vocab, 1000), p_shape, device=device, dtype=torch.long)
                else:
                    inputs[param_name] = torch.randn(
                        p_shape, device=device, requires_grad=_float_requires_grad
                    )
        else:
            axis_values = {**fixed_axis, axis_to_maximize: value}
            for param_name, _ in inputs_info:
                shape = _get_default_shape_for_param(param_name, axis_values, cfg_effective)
                dtype = _infer_input_type(param_name)
                if dtype == "integer":
                    vocab = getattr(cfg_effective, "vocab_size", None) or 50257
                    t = torch.randint(0, min(vocab, 1000), shape, device=device, dtype=torch.long)
                else:
                    t = torch.randn(shape, device=device, requires_grad=_float_requires_grad)
                inputs[param_name] = t
        return inputs

    def get_loss(output: Any) -> torch.Tensor:
        if hasattr(output, "loss") and output.loss is not None:
            return output.loss
        if isinstance(output, dict):
            if "loss" in output and output["loss"] is not None:
                return output["loss"]
            # Sum all tensor outputs
            tensors = [v for v in output.values() if isinstance(v, torch.Tensor)]
            if tensors:
                return sum(t.sum() for t in tensors)
        if isinstance(output, torch.Tensor):
            return output.sum()
        if isinstance(output, (tuple, list)):
            tensors = [v for v in output if isinstance(v, torch.Tensor)]
            if tensors:
                return sum(t.sum() for t in tensors)
        raise ValueError("Could not extract loss from model output.")

    def forward_backward_measured(
        active_model: torch.nn.Module, n: int
    ) -> Tuple[bool, Optional[str], Dict[int, int], int]:
        """Run one forward (+ optional backward); return (ok, err, peak_bytes_per_cuda_device, rss_delta)."""
        rss0 = _process_rss_bytes()
        peaks: Dict[int, int] = {}
        err: Optional[str] = None
        try:
            for d in cuda_devs:
                torch.cuda.reset_peak_memory_stats(d)
            active_model.zero_grad(set_to_none=True)
            inputs = make_inputs(n)
            if inference_only:
                with torch.no_grad():
                    out = active_model(**inputs)
                loss = get_loss(out)
            else:
                out = active_model(**inputs)
                loss = get_loss(out)
                loss.backward()
            if cuda_devs:
                for d in cuda_devs:
                    torch.cuda.synchronize(d)
                peaks = {d: int(torch.cuda.max_memory_allocated(d)) for d in cuda_devs}
            return True, None, peaks, max(0, _process_rss_bytes() - rss0)
        except Exception as e:
            err = str(e)[:200]
            if cuda_devs:
                try:
                    for d in cuda_devs:
                        torch.cuda.synchronize(d)
                    peaks = {d: int(torch.cuda.max_memory_allocated(d)) for d in cuda_devs}
                except Exception:
                    peaks = {}
            return False, err, peaks, max(0, _process_rss_bytes() - rss0)

    if use_dsl:
        desc_base = f"input_shapes search={spec.search_symbol if spec else '?'}"
    elif use_multi_shape and multi_shape_parts is not None:
        desc_base = f"input_shapes multi={multi_shape_parts!r}"
    elif use_compact_shape and compact_spec is not None:
        desc_base = f"input_shapes={compact_spec!r}"
    elif use_single_shape:
        desc_base = f"shape={single_shape}"
    else:
        desc_base = f"{axis_to_maximize} fixed={fixed_axis}"
    pbar = tqdm(range(n_attempts), total=n_attempts, desc=desc_base, position=0, leave=True)
    captured_warnings: List[str] = []
    _log_records: List[logging.LogRecord] = []

    class _CaptureHandler(logging.Handler):
        def emit(self, record: logging.LogRecord):
            _log_records.append(record)

    _log_handler = _CaptureHandler()
    successful: List[int] = []
    unsuccessful: List[int] = []
    current_value = initial_value
    success_peak_gpu_by_value: Dict[int, Dict[int, int]] = {}

    tgt_frac = min(0.99, max(0.5, float(memory_target_fraction)))
    max_mult = max(1.05, float(max_growth_multiplier))

    # mp_ctx / mp_ok were computed before the config+signature probe (same subprocess default).
    first_subprocess_run = True
    _logged_subprocess_sigkill = False
    search_t0 = time.monotonic()
    time_limit_hit = False

    def _remaining_seconds() -> Optional[float]:
        if time_limit_seconds is None:
            return None
        return max(0.0, float(time_limit_seconds) - (time.monotonic() - search_t0))

    n_gpus = 0
    for i in pbar:
        if time_limit_seconds is not None:
            if (time.monotonic() - search_t0) >= float(time_limit_seconds):
                time_limit_hit = True
                if successful:
                    tqdm.write("\n⏱ Time limit reached; returning best batch found so far.")
                else:
                    tqdm.write("\n⏱ Time limit reached; no successful trial yet.")
                break
        value_i = max(1, current_value)
        ok = False
        err_msg_str: Optional[str] = None
        n_gpus = 0
        peak_gpu_attempt: Dict[int, int] = {}
        rss_delta_attempt = 0

        result_queue: Optional[Any] = None
        proc: Optional[Any] = None

        if mp_ok and mp_ctx:
            result_queue = mp_ctx.Queue()

            def run_in_process(q, v: int, show_gpu: bool = True):
                log_msgs: List[str] = []

                class _QHandler(logging.Handler):
                    def emit(self, rec):
                        log_msgs.append(rec.getMessage())

                for _ln in ("transformers", "torch"):
                    h = _QHandler()
                    h.setLevel(logging.WARNING)
                    logr = logging.getLogger(_ln)
                    logr.addHandler(h)
                try:
                    gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
                    if show_gpu:
                        gpu_info = f"CUDA available: {torch.cuda.is_available()}"
                        if torch.cuda.is_available():
                            gpu_info += f", devices: {gpus}"
                        tqdm.write(f"[subprocess] {gpu_info}")
                    active: Optional[torch.nn.Module] = None
                    try:
                        active = get_model()
                        active = active.to(device)
                        with warnings.catch_warnings(record=True) as w:
                            warnings.simplefilter("always")
                            ok_m, err_m, peak_g, rss_d = forward_backward_measured(active, v)
                            q.put(
                                (
                                    ok_m,
                                    err_m[:60] if err_m else None,
                                    gpus,
                                    [str(x.message) for x in w],
                                    log_msgs,
                                    peak_g,
                                    rss_d,
                                )
                            )
                    finally:
                        if active is not None:
                            del active
                        gc.collect()
                        _release_memory(device, cuda_devs, aggressive=True)
                finally:
                    for _ln in ("transformers", "torch"):
                        for h in logging.getLogger(_ln).handlers[:]:
                            if isinstance(h, _QHandler):
                                logging.getLogger(_ln).removeHandler(h)

            proc = mp_ctx.Process(target=run_in_process, args=(result_queue, value_i, first_subprocess_run))
            try:
                proc.start()
                rem_join = _remaining_seconds()
                if rem_join is not None:
                    proc.join(timeout=max(0.0, rem_join))
                else:
                    proc.join()
                if proc.is_alive():
                    proc.terminate()
                    proc.join(timeout=15)
                    if proc.is_alive():
                        proc.kill()
                        proc.join(timeout=10)
                    err_msg_str = "time limit (worker terminated)"
                    ok = False
                elif proc.exitcode == 0:
                    try:
                        res = result_queue.get_nowait()
                    except queue.Empty:
                        res = (False, "empty result queue", 0, [], [], {}, 0)
                    ok = res[0]
                    err_msg_str = res[1]
                    n_gpus = res[2]
                    captured_warnings.extend(res[3] if len(res) > 3 else [])
                    captured_warnings.extend(res[4] if len(res) > 4 else [])
                    if len(res) > 5 and isinstance(res[5], dict):
                        peak_gpu_attempt = {int(k): int(v) for k, v in res[5].items()}
                    elif len(res) > 5 and isinstance(res[5], int) and res[5] > 0 and cuda_devs:
                        peak_gpu_attempt = {cuda_devs[0]: int(res[5])}
                    rss_delta_attempt = int(res[6]) if len(res) > 6 else 0
                else:
                    ec = proc.exitcode
                    err_msg_str = (
                        f"worker exit {_describe_subprocess_exitcode(ec)}"
                        if ec is not None
                        else "Process died"
                    )
                    if ec is not None and ec < 0 and not _logged_subprocess_sigkill:
                        _logged_subprocess_sigkill = True
                        tqdm.write(
                            "batch_finder: subprocess worker was terminated (often host OOM); "
                            "parent continues the search."
                        )
            except Exception:
                mp_ok = False
            else:
                first_subprocess_run = False
            finally:
                if result_queue is not None:
                    try:
                        while True:
                            try:
                                result_queue.get_nowait()
                            except queue.Empty:
                                break
                    except Exception:
                        pass
                    try:
                        result_queue.close()
                        result_queue.join_thread()
                    except Exception:
                        pass
                if proc is not None and proc.is_alive():
                    proc.terminate()
                    proc.join(timeout=30)

        if not mp_ok:
            n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
            _log_handler.setLevel(logging.WARNING)
            _log_handler.setFormatter(logging.Formatter("%(message)s"))
            for _logger_name in ("transformers", "torch"):
                log = logging.getLogger(_logger_name)
                log.addHandler(_log_handler)
            try:
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    ok = False
                    err_inner: Optional[str] = None
                    active_ip: Optional[torch.nn.Module] = None
                    try:
                        active_ip = get_model()
                        active_ip = active_ip.to(device)
                        ok, err_inner, peak_gpu_attempt, rss_delta_attempt = forward_backward_measured(
                            active_ip, value_i
                        )
                    finally:
                        if active_ip is not None:
                            del active_ip
                        gc.collect()
                        _release_memory(device, cuda_devs, aggressive=True)
                    if err_inner:
                        err_msg_str = err_inner[:60]
                    captured_warnings.extend(str(x.message) for x in w)
            finally:
                for _logger_name in ("transformers", "torch"):
                    logging.getLogger(_logger_name).removeHandler(_log_handler)

        if ok:
            successful.append(value_i)
            if peak_gpu_attempt and any(v > 0 for v in peak_gpu_attempt.values()):
                success_peak_gpu_by_value[value_i] = dict(peak_gpu_attempt)
            pf_ok: Dict[str, Any] = {
                "i": f"{i+1}/{n_attempts}",
                "value": value_i,
                "max_ok": max(successful),
                "min_fail": min(unsuccessful) if unsuccessful else None,
                "gpus": n_gpus,
                "status": "✅",
            }
            if peak_gpu_attempt and any(v > 0 for v in peak_gpu_attempt.values()):
                pf_ok["gpu_peak_mb"] = sum(peak_gpu_attempt.values()) // (1024 * 1024)
            pbar.set_postfix(**pf_ok)
        else:
            unsuccessful.append(value_i)
            pf = {"i": f"{i+1}/{n_attempts}", "value": value_i, "max_ok": max(successful) if successful else None, "min_fail": min(unsuccessful), "gpus": n_gpus, "status": "❌"}
            if err_msg_str:
                pf["err"] = err_msg_str[:40]
            pbar.set_postfix(**pf)

        _release_memory(device, cuda_devs, aggressive=not ok)
        if time_limit_seconds is not None:
            r_sleep = _remaining_seconds()
            if r_sleep is not None and r_sleep <= 0:
                time_limit_hit = True
                if successful:
                    tqdm.write("\n⏱ Time limit reached; returning best batch found so far.")
                else:
                    tqdm.write("\n⏱ Time limit reached; no successful trial yet.")
                break
            time.sleep(min(delay, r_sleep) if r_sleep is not None else delay)
        else:
            time.sleep(delay)

        t_ok = [ok]

        if t_ok and not t_ok[0] and value_i == 1:
            tqdm.write("❌ Failed at value=1; no smaller value to try.")
            return None

        max_ok = max(successful) if successful else None
        min_fail = min(unsuccessful) if unsuccessful else None
        if max_ok is not None and min_fail is not None:
            if min_fail <= max_ok + 1:
                break
            current_value = (max_ok + min_fail) // 2
        elif t_ok and t_ok[0]:
            if memory_guided:
                mult = _memory_guided_success_multiplier(
                    peak_gpu_attempt,
                    rss_delta_attempt,
                    target_fraction=tgt_frac,
                    max_multiplier=max_mult,
                    fallback=factor_up,
                )
                current_value = max(value_i + 1, int(value_i * mult))
            else:
                current_value = max(value_i + 1, int(value_i * factor_up))
        else:
            current_value = max(1, int(value_i / factor_down))

        if current_value < 1:
            break

    if captured_warnings or _log_records:
        seen = set()
        tqdm.write("")
        for msg in captured_warnings:
            if msg not in seen:
                seen.add(msg)
                tqdm.write(f"⚠ {msg}")
        for rec in _log_records:
            msg = rec.getMessage()
            if msg not in seen:
                seen.add(msg)
                tqdm.write(f"⚠ {msg}")

    if successful:
        result = max(successful)
        _world = int(os.environ.get("WORLD_SIZE", "1"))
        if _world > 1:
            _rank = int(os.environ.get("RANK", "0"))
            _sync_base = (
                distributed_sync_dir
                or os.environ.get("BATCH_FINDER_SYNC_DIR")
                or _default_distributed_sync_dir()
            )
            _sync_dir = os.path.join(
                _sync_base, "find_max_minibatch_sync", _distributed_sync_job_tag()
            )
            os.makedirs(_sync_dir, exist_ok=True)

            def _atomic_write_json(path: str, obj: Any) -> None:
                tmp = path + ".tmp"
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(obj, f)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp, path)

            if distributed_sync_dir is not None and _rank == 0:
                tqdm.write(
                    "batch_finder: using explicit distributed_sync_dir under multi-rank launch — "
                    "ensure every rank resolves the **same** absolute path (not per-process random "
                    "experiment dirs)."
                )

            _per_path = os.path.join(_sync_dir, f"_findbatch_rank_{_rank}.json")
            _atomic_write_json(
                _per_path,
                {"trial": int(result), "search_finished": True},
            )
            _agreed_path = os.path.join(_sync_dir, "_findbatch_agreed.json")
            _sync_poll_s = 4.0
            _sync_max_wait_s = 3600.0
            if _rank == 0:
                try:
                    if os.path.exists(_agreed_path):
                        os.remove(_agreed_path)
                except OSError:
                    pass
                tqdm.write(
                    f"\nbatch_finder: rank 0 finished search (best trial {result}); "
                    f"waiting for {_world} ranks to publish sync files under {_sync_dir!r} "
                    "(other ranks may still be probing — this is not a hang). "
                    f"Poll every {_sync_poll_s:.0f}s."
                )
                _wait_t0 = time.monotonic()
                _last_msg = _wait_t0
                while time.monotonic() - _wait_t0 < _sync_max_wait_s:
                    missing = [
                        r
                        for r in range(_world)
                        if not os.path.exists(
                            os.path.join(_sync_dir, f"_findbatch_rank_{r}.json")
                        )
                    ]
                    if not missing:
                        trials: List[int] = []
                        for r in range(_world):
                            with open(
                                os.path.join(_sync_dir, f"_findbatch_rank_{r}.json"),
                                encoding="utf-8",
                            ) as f:
                                row = json.load(f)
                            t = row.get("trial")
                            if t is None:
                                raise RuntimeError(
                                    f"find_max_minibatch had no successful trial on rank {r}; "
                                    "cannot agree for DDP."
                                )
                            trials.append(int(t))
                        agreed = min(trials)
                        _atomic_write_json(
                            _agreed_path, {"trial": agreed, "per_rank_trials": trials}
                        )
                        break
                    now = time.monotonic()
                    if now - _last_msg >= 30.0:
                        tqdm.write(
                            f"batch_finder: rank 0 still waiting ({now - _wait_t0:.0f}s) for "
                            f"rank file(s) {missing}."
                        )
                        _last_msg = now
                    time.sleep(_sync_poll_s)
                else:
                    raise RuntimeError(
                        "Timeout waiting for all ranks to finish find_max_minibatch (file sync)."
                    )
            else:
                tqdm.write(
                    f"\nbatch_finder: rank {_rank} finished search (best trial {result}); "
                    f"waiting for rank 0 to publish min trial under {_sync_dir!r} … "
                    f"Poll every {_sync_poll_s:.0f}s."
                )
                try:
                    _rank_mtime = os.path.getmtime(_per_path)
                except OSError:
                    _rank_mtime = 0.0
                _wait_t0 = time.monotonic()
                _last_msg = _wait_t0
                while time.monotonic() - _wait_t0 < _sync_max_wait_s:
                    if os.path.exists(_agreed_path):
                        try:
                            if os.path.getmtime(_agreed_path) >= _rank_mtime:
                                break
                        except OSError:
                            break
                    now = time.monotonic()
                    if now - _last_msg >= 30.0:
                        tqdm.write(
                            f"batch_finder: rank {_rank} still waiting ({now - _wait_t0:.0f}s) "
                            "for rank 0 to write _findbatch_agreed.json (rank 0 may still be searching)."
                        )
                        _last_msg = now
                    time.sleep(_sync_poll_s)
                else:
                    raise RuntimeError(
                        "Timeout waiting for rank 0 find_max_minibatch agreement file."
                    )
            with open(_agreed_path, encoding="utf-8") as f:
                result = int(json.load(f)["trial"])
            tqdm.write(
                f"\nMulti-rank (WORLD_SIZE={_world}): agreed trial size {result} "
                f"(min across ranks; sync dir={_sync_dir!r})."
            )
        if use_multi_shape and multi_shape_parts is not None:
            out_shapes: List[Tuple[int, ...]] = []
            for mode_i, payload_i in multi_shape_parts:
                if mode_i == "single":
                    out_shapes.append(_materialize_int_shape_slots(payload_i, result))
                else:
                    out_shapes.append(materialize_compact_numeric_shape(payload_i, result))
            final_tuple = tuple(out_shapes)
            tqdm.write(f"\n✅ Final input shapes: {final_tuple}")
            return final_tuple
        if use_compact_shape and compact_spec is not None:
            final_input_shape = materialize_compact_numeric_shape(compact_spec, result)
            tqdm.write(f"\n✅ Final input shape: {final_input_shape}")
            return final_input_shape
        if use_single_shape and single_shape is not None:
            final_input_shape = _materialize_int_shape_slots(single_shape, result)
            tqdm.write(f"\n✅ Final input shape: {final_input_shape}")
            return final_input_shape
        if use_dsl and spec is not None:
            shapes_list, _ = materialize_shapes(spec, result, fixed_axis)
            tqdm.write(f"\n✅ Max {spec.search_symbol} = {result}")
            for (name, _), sh in zip(inputs_info, shapes_list):
                tqdm.write(f"   {name}: {sh}")
            return result
        tqdm.write(f"\n✅ Max value that passed: {result}")
        return result
    tqdm.write("\n❌ No value passed without error.")
    return None
