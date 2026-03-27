"""
Batch Finder - Core functionality for finding maximum batch sizes along variable axes.
"""

import gc
import inspect
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
from typing import Optional, Callable, Dict, Any, Tuple, List, Union
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
        raw = tuple(input_shapes)
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


def _interpolated_bracket_guess(
    max_ok: int,
    min_fail: int,
    peaks_at_max: Dict[int, int],
    target_fraction: float,
    max_multiplier: float,
) -> Optional[int]:
    """VRAM linear extrapolation from per-GPU peaks at max_ok; None → use plain mid."""
    if min_fail <= max_ok + 1 or not peaks_at_max:
        return None
    if not torch.cuda.is_available():
        return None
    # Interpolation: allow large implied mult inside bracket (cap separately)
    m = _cuda_multiplier_from_peaks(peaks_at_max, target_fraction, max_multiplier=1e9)
    if m is None:
        return None
    try:
        v_est = int(max_ok * min(m, max_multiplier))
        v_est = max(max_ok + 1, min(v_est, min_fail - 1))
        if max_ok < v_est < min_fail:
            return v_est
    except Exception:
        pass
    return None


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
) -> Optional[Union[int, Tuple[int, ...]]]:
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
    4. **axis_to_maximize + fixed_axis:** when ``input_shapes`` is omitted; for multi-input
       models (e.g. HuggingFace) by symbolic axis name.

    Search strategy: by default **memory-guided** on success (GPU peak VRAM vs device total,
    optional CPU via RSS + psutil), capped by ``max_growth_multiplier`` (e.g. jump toward
    ~6× in one step when headroom allows). On failure, divide by ``factor_down``; when both
    a success and a failure bracket the answer, uses VRAM linear extrapolation when possible,
    else bisection. Set ``memory_guided=False`` to use only ``factor_up`` / ``factor_down``.

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
        input_shapes: DSL string, dict (named shapes + ``#constraints``), tuple/list of ints
            with ``-1``, or compact numeric tuple (ints + negative floats, see docstring).
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

    Returns:
        For int-only tuple/list ``input_shapes``: final shape tuple with each ``-1`` replaced
            by the max value.
        For compact numeric ``input_shapes``: final materialized shape tuple (ints only).
        For DSL string or ``axis_to_maximize``: int (max symbol or axis value).
        None if no value succeeded.
    """
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
    single_shape: Optional[Tuple[int, ...]] = shape_payload if use_single_shape else None
    compact_spec: Optional[Tuple[Union[int, float], ...]] = (
        shape_payload if use_compact_shape else None
    )
    spec: Optional[InputShapesSpec] = None
    dtype_overrides: Optional[Dict[str, str]] = None

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

    if not use_dsl and not use_single_shape and not use_compact_shape and not axis_to_maximize:
        raise ValueError("Must provide input_shapes or axis_to_maximize.")

    modifiable_axis_idxs = (
        [i for i, s in enumerate(single_shape) if s == -1]
        if (use_single_shape and single_shape)
        else []
    )
    shape_param_name = (
        inputs_info[0][0] if (use_single_shape or use_compact_shape) else None
    )

    # Build sample axis_values for size estimation
    _sample_axis_values = {**fixed_axis}
    if use_dsl and spec is not None:
        _, binds = materialize_shapes(spec, initial_value, fixed_axis)
        _sample_axis_values.update(binds)
    elif use_single_shape and single_shape is not None:
        _sample_shape = tuple(initial_value if s == -1 else s for s in single_shape)
        _sample_axis_values["batch_size"] = _sample_shape[0] if _sample_shape else initial_value
        _sample_axis_values["seq_len"] = _sample_shape[1] if len(_sample_shape) > 1 else 64
    elif use_compact_shape and compact_spec is not None:
        _sample_shape = materialize_compact_numeric_shape(compact_spec, initial_value)
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
        elif use_compact_shape and idx == 0 and compact_spec is not None:
            shape = materialize_compact_numeric_shape(compact_spec, initial_value)
        elif use_single_shape and idx == 0 and single_shape is not None:
            shape = tuple(initial_value if s == -1 else s for s in single_shape)
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
            shape = list(single_shape)
            for idx in modifiable_axis_idxs:
                shape[idx] = value
            shape = tuple(shape)
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
    n_gpus = 0
    for i in pbar:
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
                proc.join()
                if proc.exitcode == 0:
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
            if memory_guided:
                peak_at_max = success_peak_gpu_by_value.get(max_ok, {})
                guess = _interpolated_bracket_guess(
                    max_ok, min_fail, peak_at_max, tgt_frac, max_mult
                )
                current_value = guess if guess is not None else (max_ok + min_fail) // 2
            else:
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
        if use_compact_shape and compact_spec is not None:
            final_input_shape = materialize_compact_numeric_shape(compact_spec, result)
            tqdm.write(f"\n✅ Final input shape: {final_input_shape}")
            return final_input_shape
        if use_single_shape and single_shape is not None:
            final_input_shape = tuple(result if s == -1 else s for s in single_shape)
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
