"""
Batch Finder - Core functionality for finding maximum batch sizes along variable axes.
"""

import inspect
import sys
import time
import multiprocessing
import threading
from typing import Optional, Callable, Dict, Any, Tuple, List
import torch
from tqdm import tqdm


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
    fixed_dims: Optional[Dict[str, int]] = None,
    device: Optional[torch.device] = None,
    delay: float = 3.0,
    initial_value: int = 1024,
    n_attempts: int = 50,
    inference_only: bool = False,
    factor_down: float = 2.0,
    factor_up: float = 1.5,
) -> Optional[int]:
    """
    Find the maximum value for the modifiable axis that the model can process without OOM.

    Supports two modes:
    1. input_shape: Tuple with -1 for the modifiable axis, numbers for fixed dims.
       E.g. (-1, 64, 256) maximizes axis 0 with fixed (64, 256).
    2. axis_to_maximize + fixed_dims: For multi-input models (e.g. HuggingFace).

    Search strategy: unsuccessful -> value/factor_down; successful -> value*factor_up.
    Defaults: factor_down=2, factor_up=1.5 (i.e. /2 and *3/2).

    Args:
        model: PyTorch model (nn.Module or HuggingFace PreTrainedModel).
        input_shape: Tuple with -1 for variable axis, e.g. (-1, 64, 256). Takes precedence.
        axis_to_maximize: Name of axis to maximize when not using input_shape.
        fixed_dims: Dict of fixed values when using axis_to_maximize.
        device: Device to run on (default: cuda if available else cpu).
        delay: Delay in seconds between attempts.
        initial_value: Initial value to try (first attempt).
        n_attempts: Maximum attempts.
        inference_only: If True, skip forward gradients and backward pass. If False, runs full forward+backward.
        factor_down: On failure, next = value / factor_down (default 2).
        factor_up: On success, next = value * factor_up (default 1.5).

    Returns:
        Maximum value for the axis, or None if no value succeeded.
    """
    fixed_dims = fixed_dims or {}
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Detect inputs
    inputs_info = _detect_model_inputs(model)
    if not inputs_info:
        raise ValueError("Could not detect model inputs from forward signature.")

    use_input_shape = input_shape is not None
    if use_input_shape and -1 not in input_shape:
        raise ValueError("input_shape must contain at least one -1 for the modifiable axis.")
    if not use_input_shape and not axis_to_maximize:
        raise ValueError("Must provide either input_shape or axis_to_maximize.")

    modifiable_axis_idxs = [i for i, s in enumerate(input_shape) if s == -1] if (use_input_shape and input_shape) else []
    shape_param_name = inputs_info[0][0] if use_input_shape else None

    # Build sample axis_values for size estimation
    _sample_axis_values = {**fixed_dims}
    if use_input_shape:
        _sample_shape = tuple(initial_value if s == -1 else s for s in input_shape)
        _sample_axis_values["batch_size"] = _sample_shape[0] if _sample_shape else initial_value
        _sample_axis_values["seq_len"] = _sample_shape[1] if len(_sample_shape) > 1 else 64
    else:
        _sample_axis_values[axis_to_maximize] = initial_value

    tqdm.write("\n\n--- Detected inputs (type, estimated shape) ---")
    for idx, (name, _) in enumerate(inputs_info):
        itype = _infer_input_type(name)
        if use_input_shape and idx == 0:
            shape = tuple(initial_value if s == -1 else s for s in input_shape)
        else:
            shape = _get_default_shape_for_param(name, model, _sample_axis_values)
        tqdm.write(f"  {name}: {itype}, {shape}")
    tqdm.write("---")

    def make_inputs(value: int) -> Dict[str, torch.Tensor]:
        inputs = {}
        if use_input_shape:
            shape = list(input_shape)
            for idx in modifiable_axis_idxs:
                shape[idx] = value
            shape = tuple(shape)
            dtype = _infer_input_type(shape_param_name)
            if dtype == "integer":
                vocab = getattr(getattr(model, "config", None), "vocab_size", None) or 50257
                inputs[shape_param_name] = torch.randint(0, min(vocab, 1000), shape, device=device, dtype=torch.long)
            else:
                inputs[shape_param_name] = torch.randn(shape, device=device)
            # For multi-input: broadcast compatible shapes to other params (e.g. attention_mask)
            for param_name, _ in inputs_info[1:]:
                axis_values = {"batch_size": shape[0] if len(shape) >= 1 else value, "seq_len": shape[1] if len(shape) >= 2 else 64}
                p_shape = _get_default_shape_for_param(param_name, model, axis_values)
                p_dtype = _infer_input_type(param_name)
                if p_dtype == "integer":
                    vocab = getattr(getattr(model, "config", None), "vocab_size", None) or 50257
                    inputs[param_name] = torch.randint(0, min(vocab, 1000), p_shape, device=device, dtype=torch.long)
                else:
                    inputs[param_name] = torch.randn(p_shape, device=device)
        else:
            axis_values = {**fixed_dims, axis_to_maximize: value}
            for param_name, _ in inputs_info:
                shape = _get_default_shape_for_param(param_name, model, axis_values)
                dtype = _infer_input_type(param_name)
                if dtype == "integer":
                    vocab = getattr(getattr(model, "config", None), "vocab_size", None) or 50257
                    t = torch.randint(0, min(vocab, 1000), shape, device=device, dtype=torch.long)
                else:
                    t = torch.randn(shape, device=device)
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

    desc_base = f"shape={input_shape}" if use_input_shape else f"{axis_to_maximize} fixed={fixed_dims}"
    pbar = tqdm(range(n_attempts), total=n_attempts, desc=desc_base, position=0, leave=False)
    successful: List[int] = []
    unsuccessful: List[int] = []
    current_value = initial_value

    # Subprocess + fork breaks CUDA context in child. Use in-process when GPU.
    use_subprocess = sys.platform != "win32" and not torch.cuda.is_available()
    first_subprocess_run = True
    for i in pbar:
        value_i = max(1, current_value)
        ok = False
        err_msg_str: Optional[str] = None

        if use_subprocess:
            result_queue = multiprocessing.Queue()

            def run_in_process(q, v: int, show_gpu: bool = True):
                if show_gpu:
                    gpu_info = f"CUDA available: {torch.cuda.is_available()}"
                    if torch.cuda.is_available():
                        gpu_info += f", devices: {torch.cuda.device_count()}"
                    tqdm.write(f"[subprocess] {gpu_info}")
                try:
                    forward_and_backward(v)
                    q.put((True, None))
                except Exception as e:
                    q.put((False, str(e)[:60]))

            proc = multiprocessing.Process(target=run_in_process, args=(result_queue, value_i, first_subprocess_run))
            try:
                proc.start()
                proc.join()
                if proc.exitcode == 0:
                    ok, err_msg_str = result_queue.get_nowait()
                else:
                    err_msg_str = f"Killed (exitcode {proc.exitcode})" if proc.exitcode else "Process died"
            except Exception:
                use_subprocess = False
            else:
                first_subprocess_run = False

        if not use_subprocess:
            try:
                forward_and_backward(value_i)
                ok = True
            except Exception as e:
                err_msg_str = str(e)[:60]

        if ok:
            successful.append(value_i)
            pbar.set_postfix(
                i=f"{i+1}/{n_attempts}", value=value_i,
                max_ok=max(successful), min_fail=min(unsuccessful) if unsuccessful else None,
                status="✅",
            )
        else:
            unsuccessful.append(value_i)
            pf = {"i": f"{i+1}/{n_attempts}", "value": value_i, "max_ok": max(successful) if successful else None, "min_fail": min(unsuccessful), "status": "❌"}
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

        if t_ok and t_ok[0]:
            current_value = max(value_i + 1, int(value_i * factor_up))
        else:
            current_value = max(1, int(value_i / factor_down))

        if current_value < 1:
            break

    if successful:
        result = max(successful)
        tqdm.write(f"\n\n✅ Max value that passed: {result}")
        return result
    tqdm.write("\n\n❌ No value passed without error.")
    return None
