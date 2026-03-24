# 🔍 Batch Finder

**Find the maximum value for any dimension your PyTorch models can handle without running out of memory.**

Batch Finder automatically detects your model's inputs (type and shape), fixes the dimensions you specify, and finds the maximum value for the remaining axis using a configurable search strategy.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## ✨ Features

- 🎯 **Single unified API** – One function `find_max_minibatch` for all cases
- 🔍 **Automatic input detection** – Infers input names, types (int/float), and shapes from the model
- 📐 **Flexible shape specification** – **`input_shapes`** as tuple/list (single tensor), DSL **string** (multi-tensor), or `axis_to_maximize` + `fixed_axis`
- 🚀 **Inference or full backward** – Test with or without gradients
- ⚙️ **Configurable search** – Customize `factor_down`, `factor_up`, `n_attempts`, `initial_value`
- 🛡️ **Safe testing** – Error handling, memory cleanup, returns `None` if fails at value 1
- 📊 **Progress tracking** – tqdm progress bar with status in postfix

## 📦 Installation

```bash
pip install batch-finder
```

Or install from source:

```bash
git clone https://github.com/yourusername/batch-finder.git
cd batch-finder
pip install -e .
```

## 🚀 Quick Start

### Mode 1: `input_shapes` as tuple or list (single-input models)

Use a tuple or list of ints with `-1` for the axis to maximize and numbers for fixed dimensions:

```python
from batch_finder import find_max_minibatch

model = MyModel()

# Maximize axis 0, fix (64, 256)
max_val = find_max_minibatch(model=model, input_shapes=(-1, 64, 256))

# Maximize axis 2, fix (4, 8)
max_val = find_max_minibatch(model=model, input_shapes=(4, 8, -1))

# Multiple -1: same value for all variable axes
max_val = find_max_minibatch(model=model, input_shapes=(-1, 4, -1, 16))
```

**Compact numeric tuple (optional, still single-tensor `input_shapes`):** mix **int** sizes with **negative floats** to tie an axis to the searched size without a DSL string. You still need **exactly one integer `-1`** marking the binary‑searched axis. A **negative float** `d` means that dimension’s size is `round(abs(d) * s)`, where `s` is the current trial value for the `-1` axis (e.g. `-1.5` → `1.5×` that size).

```python
# Search axis 0; axis 2 is always round(1.5 * axis_0)
max_val = find_max_minibatch(model=model, input_shapes=(-1, 4, -1.5, 16))
# Same idea as the string DSL "… t=1.5b, b=-1" when only one tensor is involved.
```

Use **integer** `-1` for the search axis, not `-1.0` (a float is treated as a scale factor). Multi-tensor symbolic layouts still use the **string** (or dict) DSL.

### Mode 2: `axis_to_maximize` + `fixed_axis` (multi-input models, e.g. HuggingFace)

```python
from transformers import AutoModelForCausalLM
from batch_finder import find_max_minibatch

model = AutoModelForCausalLM.from_pretrained("distilgpt2")
max_batch = find_max_minibatch(
    model=model,
    axis_to_maximize="batch_size",
    fixed_axis={"seq_len": 32},
)
print(f"Max batch size: {max_batch}")
```

### Mode 3: `input_shapes` (compact multi-tensor string)

For models whose `forward` takes **several tensors**, pass **`input_shapes` as a string**: one compact DSL with a shape tuple **per argument** (same order as `forward`), **names** for dimensions that must match across tensors, optional **constraints** between names, and **exactly one** searched dimension with `name=-1`.

**Layout:**  
`(dims...), (dims...), ... , constraint, constraint, ...`

- **Dimensions:** non‑negative integers, or identifiers (`b`, `t`, …). The same name in different positions ties those axes to the **same** size when materialized.
- **Search:** one assignment must be `symbol=-1`. Batch Finder binary‑searches that symbol (e.g. maximize `b`).
- **Constraints:** `name=rhs` where `rhs` can be an integer, another name, `coef*name`, `coef` stuck to a name (e.g. `1.5b`), or `name*name`. Values that are not integers are **rounded** to the nearest integer for tensor sizes.

```python
import torch
from batch_finder import find_max_minibatch

class MyModel(torch.nn.Module):
    def forward(self, x, y):
        # x: (23, b, t, 45), y: (b, t, 12) — example layout
        ...

model = MyModel()
max_b = find_max_minibatch(
    model=model,
    input_shapes="(23, b, t, 45),(b, t, 12), t=1.5b, b=-1",
)
# Searches b; sets t = round(1.5 * b) each trial. Returns max b (int).
```

Mutually exclusive with tuple/list single-tensor mode and with `axis_to_maximize`. Pass `input_shapes=` as a **keyword** argument.

### Mode 3b: `input_shapes` as a dict (named arguments)

Same semantics as the string DSL, but each `forward` parameter is keyed by name. Values are shape strings, optionally followed by `, int` or `, float` for tensor dtype. Use the key `"#constraints"` for the constraint list (must include exactly one `symbol=-1`).

```python
max_b = find_max_minibatch(
    model=model,
    input_shapes={
        "input_ids": "(b, t), int",
        "attention_mask": "(b, t), int",
        "input_ids_encoder": "(d, b, t), int",
        "attention_mask_encoder": "(d, b, t), int",
        "labels": "(b, t)",
        "#constraints": "t=2b, b=-1",
    },
)
```

Import `CONSTRAINTS_KEY` from `batch_finder` (value `"#constraints"`) if you prefer not to spell the string. `FINDER_CONSTRAINTS_KEY` is an alias for the same value.

### Custom search parameters

```python
max_val = find_max_minibatch(
    model=model,
    input_shapes=(-1, 128, 512),
    initial_value=8,
    n_attempts=30,
    factor_down=3.0,   # divide by 3 on failure
    factor_up=2.0,     # multiply by 2 on success
)
```

## 📖 API Reference

### `find_max_minibatch(model, ...)`

Find the maximum value for the modifiable axis without OOM.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `torch.nn.Module` | – | PyTorch or HuggingFace model |
| `input_shapes` | `str` \| `dict` \| tuple/list | `None` | **String:** multi-tensor DSL (Mode 3). **Dict:** named shapes + `"#constraints"` (Mode 3b). **Tuple/list:** single first `forward` arg — all-**int** with `-1`, or **compact numeric** with one int `-1` and negative **floats** for scaled axes (e.g. `-1.5` → `1.5×` search). Mutually exclusive with `axis_to_maximize` |
| `axis_to_maximize` | `str` | `None` | Axis name when `input_shapes` is omitted, e.g. `"batch_size"` |
| `fixed_axis` | `Dict[str, int]` | `{}` | Fixed values, e.g. `{"seq_len": 128}` |
| `device` | `torch.device` | auto | Device to run on |
| `delay` | `float` | `None`→auto | Auto: `3.0` s on CUDA, `0.75` s on CPU/MPS (pass a number to override) |
| `initial_value` | `int` | `None`→auto | Auto: `512` CUDA, `64` MPS, `32` CPU (pass a number to override) |
| `n_attempts` | `int` | `50` | Maximum attempts |
| `inference_only` | `bool` | `False` | If `True`, no gradients. If `False`, full forward+backward. |
| `factor_down` | `float` | `2.0` | On failure: `next = value / factor_down` |
| `factor_up` | `float` | `2.0` | Used when `memory_guided=False` for success steps |
| `memory_guided` | `bool` | `True` | Use peak GPU VRAM (and optional CPU via `psutil`) to pick the next size; cap with `max_growth_multiplier` |
| `memory_target_fraction` | `float` | `0.88` | Target peak VRAM as a fraction of total GPU memory when extrapolating |
| `max_growth_multiplier` | `float` | `6.0` | Max single-step increase after success (e.g. jump toward 6× when headroom allows) |
| `cuda_mem_devices` | `int`, list, or `"all"` | `None` | Which GPUs to read peak memory on; default = model device only. Use `"all"` or `[0,1,…]` for multi-GPU; **bottleneck** (tightest headroom) sets the next step |
| `use_subprocess` | `bool` | `None` | Per-attempt child on Linux/macOS (default on) so OOM usually kills only the worker; `False` or env `BATCH_FINDER_SUBPROCESS=0` for in-process |

**Returns:** `Tuple[int, ...]` (when using tuple/list `input_shapes`), `int` (when using DSL string or `axis_to_maximize`), or `None` if nothing succeeded.

**Modes:**
- Provide `input_shapes` as tuple/list: uses first input param; `-1` = variable axis (all ints), or compact numeric tuple with negative floats for scaled axes.
- Provide `input_shapes` as string: one shape tuple per `forward` tensor; symbolic names + `symbol=-1` + optional constraints.
- Provide `input_shapes` as dict: same as string DSL, with keys = parameter names and `"#constraints"` for constraints.
- Provide `axis_to_maximize` + `fixed_axis`: builds inputs from detected params and conventions.

**Example output** (axis_to_maximize + fixed_axis):

```
--- Detected inputs (type, estimated shape) ---
  input_ids: integer, (32, 64)
  attention_mask: integer, (32, 64)
---

batch_size fixed={'seq_len': 32}: 100%|████████████████████| 22/50 [01:26<00:00,  3.9s/it, gpus=1, i=22/50, max_ok=1919, min_fail=1920, status=✅, value=1919]

✅ Max value that passed: 1919
```

## 🔧 How It Works

1. **Input detection** – Uses `inspect.signature` on `model.forward` to find input names.
2. **Type inference** – Integer for `*ids`, `*mask`, `labels`; float for others.
3. **Shape estimation** – From model (Linear.in_features, config, etc.) and param-name conventions.
4. **Search** – On success: try `value * factor_up`. On failure: try `value / factor_down`. Stops when value 1 fails or `n_attempts` reached.
5. **Loss** – Uses `output.loss` if present, else sum of all output tensors.

## ⚠️ Important Notes

- **Memory:** Use conservative `initial_value` on limited GPU memory.
- **Time:** Use `inference_only=True` for faster runs.
- **Training:** Use `inference_only=False` to stress-test with backward pass.
- **Value 1:** If the run fails at value 1, the function returns `None` (no smaller value).
- **OOM and your process:** PyTorch **`torch.cuda.OutOfMemoryError`**, **`MemoryError`**, and similar allocator errors are **caught**; the search **continues** with a smaller size. On **Linux/macOS**, each attempt runs in a **subprocess by default** so if the worker is **SIGKILL**’d by the kernel (hard OOM), the **parent** usually keeps running and tries again. On **Windows**, or if you set **`use_subprocess=False`** or **`BATCH_FINDER_SUBPROCESS=0`**, work runs **in-process**—a **hard kill** of the **main** process can still stop the script (nothing can catch SIGKILL). After a failed attempt, the code runs **`gc.collect()`** and device cache clears where applicable.
- **CPU vs GPU defaults:** If you omit `initial_value` and `delay`, the library picks **smaller starting sizes and shorter pauses on CPU** (and MPS); CUDA keeps larger defaults. Override anytime.
- **Login nodes / memory:** If subprocess per attempt is too heavy on a shared CPU node, set **`BATCH_FINDER_SUBPROCESS=0`** and use smaller `initial_value` / `inference_only=True`.
- **Memory-guided steps:** After each successful forward/backward, the library reads **peak GPU memory** (`torch.cuda.max_memory_allocated`) vs **device total** to estimate how much larger the batch can be in one step (capped by **`max_growth_multiplier`**, default 6). Install **`psutil`** for a rough **CPU RAM** headroom hint as well. When a failing size brackets the answer, it uses **VRAM linear extrapolation** from the last good run when possible, otherwise bisection. Set **`memory_guided=False`** to use only `factor_up` / `factor_down`.
- **Multi-GPU:** Pass **`cuda_mem_devices="all"`** or **`[0, 1, …]`** so peaks are read on each GPU; the next batch size uses the **smallest** safe multiplier across them (bottleneck). Default is only the tensor **`device`**’s index.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Made with ❤️ for the PyTorch community**
