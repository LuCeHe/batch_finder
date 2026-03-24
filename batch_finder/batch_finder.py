"""
Batch Finder - Core functionality for finding maximum batch sizes along variable axes.
"""

import inspect
import logging
import sys
import time
import warnings
import multiprocessing
import threading
from typing import Optional, Callable, Dict, Any, Tuple, List, Union
import torch
from tqdm import tqdm

from .input_shapes import InputShapesSpec, materialize_shapes, parse_input_shapes, collect_symbols_in_shapes


# Parameters to skip (not tensor inputs we can synthesize)
_SKIP_PARAMS = {
    "past_key_values", "head_mask", "encoder_hidden_states",
    "encoder_attention_mask", "cross_attn_head_mask", "inputs_embeddings",
    "inputs_embeds",  # HF: conflicts with input_ids
    "use_cache", "output_attentions", "output_hidden_states", "return_dict",
    "cache_position",  # HF: internal position tracking for KV cache
    "logits_to_keep",  # HF: internal indexing, expects specific format
}


def _detect_model_inputs(model: torch.nn.Module) -> List[Tuple[str, Any]]:
    """Detect input parameters from the model's forward signature."""
    try:
        sig = inspect.signature(model.forward)
    except (ValueError, TypeError):
        return []

    params = []
    for name, param in sig.parameters.items():
        if name in ("self", "args", "kwargs"):
            continue
        if name in _SKIP_PARAMS:
            continue
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        params.append((name, param))

    return params


def _infer_input_type(param_name: str) -> str:
    """Infer whether an input expects integer or float tensors."""
    name_lower = param_name.lower()
    if any(kw in name_lower for kw in ("ids", "mask", "labels", "indices", "token_type")):
        return "integer"
    # Default for hidden_states, pixel_values, embeddings, x, input, etc.
    return "float"


def _get_hidden_size(model: torch.nn.Module) -> int:
    """Infer hidden/embedding size from model."""
    cfg = getattr(model, "config", None)
    if cfg is not None:
        for attr in ("hidden_size", "embed_dim", "d_model"):
            v = getattr(cfg, attr, None)
            if v is not None:
                return v
    for m in model.modules():
        if hasattr(m, "in_features"):
            return m.in_features
        if hasattr(m, "embedding_dim"):
            return m.embedding_dim
    return 768


def _estimate_shape_from_model(model: torch.nn.Module, param_name: str) -> Optional[Tuple[int, ...]]:
    """Estimate input shape from model structure (layers, config)."""
    name_lower = param_name.lower()
    cfg = getattr(model, "config", None)

    # HuggingFace / config-driven
    if cfg is not None:
        if "input_ids" in name_lower or "attention_mask" in name_lower or "token_type" in name_lower:
            seq = getattr(cfg, "max_position_embeddings", None) or getattr(cfg, "max_seq_length", None) or 512
            return (2, seq)  # (batch, seq)
        if "pixel" in name_lower:
            h = getattr(cfg, "image_size", 224)
            if not isinstance(h, int) and hasattr(cfg, "vision_config"):
                vc = getattr(cfg, "vision_config", None)
                h = getattr(vc, "image_size", 224) if vc else 224
            h = h if isinstance(h, int) else 224
            return (2, 3, h, h)

    # From first consuming layer
    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            return (2, 64, m.in_features)  # (batch, seq, hidden)
        if isinstance(m, torch.nn.Embedding):
            return (2, 64)  # (batch, seq) for token ids
        if isinstance(m, (torch.nn.Conv1d, torch.nn.Conv2d)):
            if hasattr(m, "in_channels"):
                return (2, m.in_channels, 224, 224) if isinstance(m, torch.nn.Conv2d) else (2, m.in_channels, 64)
    return None


def _get_default_shape_for_param(
    param_name: str,
    model: torch.nn.Module,
    axis_values: Dict[str, int],
) -> Tuple[int, ...]:
    """Get a default shape from model structure + axis values + conventions."""
    name_lower = param_name.lower()

    batch = axis_values.get("batch_size", 2)
    seq_len = axis_values.get("seq_len", 64)
    n_docs = axis_values.get("n_docs", 1)
    hidden = _get_hidden_size(model)

    # RAG-style: (n_docs, batch, seq) or similar
    if "encoder" in name_lower and "input" in name_lower:
        return (n_docs, batch, seq_len)
    if "encoder" in name_lower and "mask" in name_lower:
        return (n_docs, batch, seq_len)

    # Standard: (batch, seq) or (batch, seq, hidden)
    if "input_ids" in name_lower or "attention_mask" in name_lower or "token_type" in name_lower:
        return (batch, seq_len)
    if "position_ids" in name_lower:
        return (batch, seq_len)
    if "labels" in name_lower or "label" in name_lower:
        return (batch, seq_len)

    # pixel_values: (batch, channels, height, width)
    if "pixel" in name_lower:
        return (batch, 3, 224, 224)

    # Generic x or input: infer from model (Linear.in_features, etc.) or (batch, seq, hidden)
    if name_lower in ("x", "input", "hidden_states", "inputs_embeddings"):
        est = _estimate_shape_from_model(model, param_name)
        if est is not None and len(est) >= 3:
            return (batch, seq_len, est[2])  # use inferred last dim
        return (batch, seq_len, hidden)

    # Fallback: try model-based estimate
    est = _estimate_shape_from_model(model, param_name)
    if est is not None:
        out = list(est)
        if len(out) >= 1:
            out[0] = batch
        if len(out) >= 2:
            out[1] = seq_len
        return tuple(out)

    return (batch, seq_len)


def find_max_minibatch(
    model: torch.nn.Module,
    input_shape: Optional[Tuple[int, ...]] = None,
    axis_to_maximize: Optional[str] = None,
    fixed_axis: Optional[Dict[str, int]] = None,
    device: Optional[torch.device] = None,
    delay: float = 3.0,
    initial_value: int = 512,
    n_attempts: int = 50,
    inference_only: bool = False,
    factor_down: float = 2.0,
    factor_up: float = 2.0,
    input_shapes: Optional[str] = None,
) -> Optional[Union[int, Tuple[int, ...]]]:
    """
    Find the maximum value for the modifiable axis that the model can process without OOM.

    Supports three modes:
    1. input_shapes: DSL string with one shape tuple per forward tensor argument, shared
       symbols, optional constraints (e.g. ``t=1.5b``), and exactly one ``symbol=-1`` for
       the dimension to maximize.
    2. input_shape: Tuple with -1 for the modifiable axis, numbers for fixed dims.
       E.g. (-1, 64, 256) maximizes axis 0 with fixed (64, 256).
    3. axis_to_maximize + fixed_axis: For multi-input models (e.g. HuggingFace).

    Search strategy: unsuccessful -> value/factor_down; successful -> value*factor_up.
    Defaults: factor_down=2, factor_up=2 (i.e. /2 and *2).

    Args:
        model: PyTorch model (nn.Module or HuggingFace PreTrainedModel).
        input_shape: Tuple with -1 for variable axis, e.g. (-1, 64, 256). Mutually exclusive
            with input_shapes.
        axis_to_maximize: Name of axis to maximize when not using input_shape/input_shapes.
        fixed_axis: Dict of fixed symbol or axis values (DSL extra symbols or HF-style keys).
        device: Device to run on (default: cuda if available else cpu).
        delay: Delay in seconds between attempts.
        initial_value: Initial value to try (first attempt).
        n_attempts: Maximum attempts.
        inference_only: If True, skip forward gradients and backward pass. If False, runs full forward+backward.
        factor_down: On failure, next = value / factor_down (default 2).
        factor_up: On success, next = value * factor_up (default 2).
        input_shapes: Multi-tensor symbolic shape DSL (mutually exclusive with input_shape).

    Returns:
        When using input_shape: final_input_shape (tuple with max value(s) filled in for -1).
        When using input_shapes or axis_to_maximize: maximum value for the searched symbol / axis (int).
        None if no value succeeded.
    """
    fixed_axis = fixed_axis or {}
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    if input_shapes is not None and input_shape is not None:
        raise ValueError("Provide only one of input_shapes or input_shape.")
    if input_shapes is not None and axis_to_maximize is not None:
        raise ValueError("Do not combine input_shapes with axis_to_maximize.")

    # Detect inputs
    inputs_info = _detect_model_inputs(model)
    if not inputs_info:
        raise ValueError("Could not detect model inputs from forward signature.")

    use_dsl = input_shapes is not None
    use_input_shape = input_shape is not None and not use_dsl
    spec: Optional[InputShapesSpec] = None

    if use_dsl:
        spec = parse_input_shapes(input_shapes)
        if len(spec.shapes) != len(inputs_info):
            raise ValueError(
                f"input_shapes has {len(spec.shapes)} tensor(s) but forward has {len(inputs_info)} tensor argument(s)."
            )
        syms = collect_symbols_in_shapes(spec)
        if spec.search_symbol not in syms:
            raise ValueError(
                f"search symbol {spec.search_symbol!r} must appear in at least one shape tuple."
            )
    elif use_input_shape:
        if -1 not in input_shape:
            raise ValueError("input_shape must contain at least one -1 for the modifiable axis.")
    elif not axis_to_maximize:
        raise ValueError("Must provide input_shapes, input_shape, or axis_to_maximize.")

    modifiable_axis_idxs = [i for i, s in enumerate(input_shape) if s == -1] if (use_input_shape and input_shape) else []
    shape_param_name = inputs_info[0][0] if use_input_shape else None

    # Build sample axis_values for size estimation
    _sample_axis_values = {**fixed_axis}
    if use_dsl and spec is not None:
        _, binds = materialize_shapes(spec, initial_value, fixed_axis)
        _sample_axis_values.update(binds)
    elif use_input_shape:
        _sample_shape = tuple(initial_value if s == -1 else s for s in input_shape)
        _sample_axis_values["batch_size"] = _sample_shape[0] if _sample_shape else initial_value
        _sample_axis_values["seq_len"] = _sample_shape[1] if len(_sample_shape) > 1 else 64
    else:
        _sample_axis_values[axis_to_maximize] = initial_value

    tqdm.write("\n\n--- Detected inputs (type, estimated shape) ---")
    for idx, (name, _) in enumerate(inputs_info):
        itype = _infer_input_type(name)
        if use_dsl and spec is not None:
            shapes_list, _ = materialize_shapes(spec, initial_value, fixed_axis)
            shape = shapes_list[idx]
        elif use_input_shape and idx == 0:
            shape = tuple(initial_value if s == -1 else s for s in input_shape)
        else:
            shape = _get_default_shape_for_param(name, model, _sample_axis_values)
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
                dtype = _infer_input_type(param_name)
                if dtype == "integer":
                    vocab = getattr(getattr(model, "config", None), "vocab_size", None) or 50257
                    inputs[param_name] = torch.randint(0, min(vocab, 1000), shape, device=device, dtype=torch.long)
                else:
                    inputs[param_name] = torch.randn(
                        shape, device=device, requires_grad=_float_requires_grad
                    )
        elif use_input_shape:
            shape = list(input_shape)
            for idx in modifiable_axis_idxs:
                shape[idx] = value
            shape = tuple(shape)
            dtype = _infer_input_type(shape_param_name)
            if dtype == "integer":
                vocab = getattr(getattr(model, "config", None), "vocab_size", None) or 50257
                inputs[shape_param_name] = torch.randint(0, min(vocab, 1000), shape, device=device, dtype=torch.long)
            else:
                inputs[shape_param_name] = torch.randn(
                    shape, device=device, requires_grad=_float_requires_grad
                )
            # For multi-input: broadcast compatible shapes to other params (e.g. attention_mask)
            for param_name, _ in inputs_info[1:]:
                axis_values = {"batch_size": shape[0] if len(shape) >= 1 else value, "seq_len": shape[1] if len(shape) >= 2 else 64}
                p_shape = _get_default_shape_for_param(param_name, model, axis_values)
                p_dtype = _infer_input_type(param_name)
                if p_dtype == "integer":
                    vocab = getattr(getattr(model, "config", None), "vocab_size", None) or 50257
                    inputs[param_name] = torch.randint(0, min(vocab, 1000), p_shape, device=device, dtype=torch.long)
                else:
                    inputs[param_name] = torch.randn(
                        p_shape, device=device, requires_grad=_float_requires_grad
                    )
        else:
            axis_values = {**fixed_axis, axis_to_maximize: value}
            for param_name, _ in inputs_info:
                shape = _get_default_shape_for_param(param_name, model, axis_values)
                dtype = _infer_input_type(param_name)
                if dtype == "integer":
                    vocab = getattr(getattr(model, "config", None), "vocab_size", None) or 50257
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

    def forward_and_backward(n: int):
        model.zero_grad(set_to_none=True)
        inputs = make_inputs(n)
        if inference_only:
            with torch.no_grad():
                out = model(**inputs)
            loss = get_loss(out)
        else:
            out = model(**inputs)
            loss = get_loss(out)
            loss.backward()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    if use_dsl:
        desc_base = f"input_shapes search={spec.search_symbol if spec else '?'}"
    elif use_input_shape:
        desc_base = f"shape={input_shape}"
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

    # Use subprocess on Linux for OOM protection. Spawn (not fork) to avoid broken CUDA context.
    use_subprocess = sys.platform != "win32"
    mp_ctx = multiprocessing.get_context("spawn") if use_subprocess else None
    first_subprocess_run = True
    n_gpus = 0
    for i in pbar:
        value_i = max(1, current_value)
        ok = False
        err_msg_str: Optional[str] = None
        n_gpus = 0

        if use_subprocess and mp_ctx:
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
                    with warnings.catch_warnings(record=True) as w:
                        warnings.simplefilter("always")
                        try:
                            forward_and_backward(v)
                            q.put((True, None, gpus, [str(x.message) for x in w], log_msgs))
                        except Exception as e:
                            q.put((False, str(e)[:60], gpus, [str(x.message) for x in w], log_msgs))
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
                    res = result_queue.get_nowait()
                    ok = res[0]
                    err_msg_str = res[1]
                    n_gpus = res[2]
                    captured_warnings.extend(res[3] if len(res) > 3 else [])
                    captured_warnings.extend(res[4] if len(res) > 4 else [])
                else:
                    err_msg_str = f"Killed (exitcode {proc.exitcode})" if proc.exitcode else "Process died"
            except Exception:
                use_subprocess = False
            else:
                first_subprocess_run = False

        if not use_subprocess:
            n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
            _log_handler.setLevel(logging.WARNING)
            _log_handler.setFormatter(logging.Formatter("%(message)s"))
            for _logger_name in ("transformers", "torch"):
                log = logging.getLogger(_logger_name)
                log.addHandler(_log_handler)
            try:
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    try:
                        forward_and_backward(value_i)
                        ok = True
                    except Exception as e:
                        err_msg_str = str(e)[:60]
                    captured_warnings.extend(str(x.message) for x in w)
            finally:
                for _logger_name in ("transformers", "torch"):
                    logging.getLogger(_logger_name).removeHandler(_log_handler)

        if ok:
            successful.append(value_i)
            pbar.set_postfix(
                i=f"{i+1}/{n_attempts}", value=value_i,
                max_ok=max(successful), min_fail=min(unsuccessful) if unsuccessful else None,
                gpus=n_gpus, status="✅",
            )
        else:
            unsuccessful.append(value_i)
            pf = {"i": f"{i+1}/{n_attempts}", "value": value_i, "max_ok": max(successful) if successful else None, "min_fail": min(unsuccessful), "gpus": n_gpus, "status": "❌"}
            if err_msg_str:
                pf["err"] = err_msg_str[:40]
            pbar.set_postfix(**pf)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
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
        if use_input_shape:
            final_input_shape = tuple(result if s == -1 else s for s in input_shape)
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
