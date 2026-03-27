# 🔍 Batch Finder



**Find the maximum value for any dimension your PyTorch models can handle without running out of memory.**

Batch Finder detects your model’s inputs (types and shapes), fixes the sizes you choose, and searches for the largest value that still fits in memory.


[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)



## ✨ Features

- 🎯 **One API** – `find_max_minibatch` for all workflows
- 🔍 **Explicit inputs** – Names, dtypes, and rough shapes from `input_shapes` / `forward_params`, plus `get_model().config` when available
- 📐 **Shapes** – Tuple/list for a single tensor, text or dict when `forward` takes several tensors, or `axis_to_maximize` + `fixed_axis`
- 🚀 **Inference or training** – With or without backward
- ⚙️ **Tunable search** – `factor_down`, `factor_up`, `n_attempts`, `initial_value`
- 🛡️ **Safe runs** – Cleans up after failures; returns `None` if even size `1` fails
- 📊 **Progress** – tqdm with status in the bar

## 📦 Installation



```bash
pip install batch-finder
```

Or from source:

```bash
git clone https://github.com/yourusername/batch-finder.git
cd batch-finder
pip install -e .
```

## 🚀 Quick Start

### One tensor: tuple or list

Use **negative integers** `-1` on each axis you want to maximize (the search tries a single trial size each step). In an **all-integer** tuple, every `-1` position shares that same trial size. Use positive integers for fixed dimensions.

If the tuple mixes integers and **negative floats**, you are in **compact numeric** mode: there must be **exactly one** integer `-1` (the searched axis). Any other dimension given as a **negative float** `-x` is sized as `round(|x| × trial)`, where `trial` is the current value on the `-1` axis—so `|x|` is the proportion you want between that axis and the searched axis (e.g. `-1.5` keeps that dim about 1.5× the trial size). Do not use `-1.0` for the search axis (use integer `-1`).

```python
from batch_finder import find_max_minibatch

def get_model():
    return MyModel()

# Maximize axis 0; other dims fixed
max_val = find_max_minibatch(get_model, input_shapes=(-1, 64, 256))

# Maximize axis 2
max_val = find_max_minibatch(get_model, input_shapes=(4, 8, -1))

# One integer -1 (searched axis) + negative float: other dims scale by |float| vs. that trial
max_val = find_max_minibatch(get_model, input_shapes=(-1, 4, -1.5, 16))
```

For several input tensors, use a string or dict below.

### HuggingFace-style: `axis_to_maximize` + `fixed_axis`


```python
from transformers import AutoModelForCausalLM
from batch_finder import find_max_minibatch

def get_model():
    return AutoModelForCausalLM.from_pretrained("distilgpt2")

# ``forward_params`` defaults to GPT2-style causal LM names; pass your own list for other architectures.
max_batch = find_max_minibatch(
    get_model,
    axis_to_maximize="batch_size",
    fixed_axis={"seq_len": 32},
)
print(f"Max batch size: {max_batch}")
```

### Several tensors: one string

When `forward` takes **multiple tensors**, pass **`input_shapes` as text**: one `(…)` group per argument (same order as `forward`), short names for axes that must match, optional equations between names, and **exactly one** `name=-1` for the size you search.



**Pattern:** `(dims…), (dims…), …` then optional `name=value` rules.

- Dimensions: numbers, or names like `b`, `t` that repeat across tensors.
- One name must be set to `-1` (that is the searched size).
- Rules like `t=1.5b` tie sizes together (non-integers are rounded for tensor shapes).



```python
import torch
from batch_finder import find_max_minibatch

class MyModel(torch.nn.Module):
    def forward(self, x, y):
        # x: (23, b, t, 45), y: (b, t, 12)

        ...

def get_model():
    return MyModel()

max_b = find_max_minibatch(
    get_model,
    input_shapes="(23, b, t, 45),(b, t, 12), t=1.5b, b=-1",
    forward_params=["x", "y"],
)
```

Do not combine this with tuple/list single-tensor mode or with `axis_to_maximize`. Pass `input_shapes=` as a keyword.

### Several tensors: dict (named arguments)

Same idea as the string form, but keys match `forward` parameters. Put shared rules in `"#constraints"` (must include exactly one `symbol=-1`). Values are shape text, optionally with `, int` or `, float` for dtype.


```python
def get_model():
    return MyModel()

max_b = find_max_minibatch(
    get_model,
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

Use the literal `"#constraints"` for the rules entry, or import `FINDER_CONSTRAINTS_KEY` from `batch_finder` to avoid typos (`CONSTRAINTS_KEY` is the same string, kept for compatibility).

### Custom search parameters

```python
def get_model():
    return MyModel()

max_val = find_max_minibatch(
    get_model,
    input_shapes=(-1, 128, 512),
    initial_value=8,
    n_attempts=30,
    factor_down=3.0,   # divide by 3 on failure
    factor_up=2.0,     # multiply by 2 on success
)
```

## 📖 API Reference

### `find_max_minibatch(...)`

Find the largest workable size for the free axis without OOM.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `get_model` | `Callable[[], nn.Module]` | (required) | **First argument.** Fresh module per attempt. Parent calls `get_model()` **once** first, keeps only `.config` (if any) for vocab/shape hints, then drops weights. Picklable under `spawn` (module-level function, not `lambda`) |
| `forward_params` | `Sequence[str]` | `None` | When `input_shapes` is not a dict: ordered tensor names for `forward`. With **`axis_to_maximize`** only, defaults to **`DEFAULT_FORWARD_PARAMS_CAUSAL_LM`** (GPT2-style). Override or import that constant to tweak. |
| `input_shapes` | `str` \| `dict` \| tuple/list | `None` | **String:** several tensors, shared names and rules. **Dict:** named shapes + `"#constraints"`. **Tuple/list:** first tensor only — all ints with `-1`, or one int `-1` plus negative floats (e.g. `-1.5` → 1.5× the trial size on that axis). Not used together with `axis_to_maximize` |
| `axis_to_maximize` | `str` | `None` | Name of the axis when `input_shapes` is omitted, e.g. `"batch_size"` |
| `fixed_axis` | `Dict[str, int]` | `{}` | Fixed sizes, e.g. `{"seq_len": 128}` |
| `device` | `torch.device` | auto | Device to run on |
| `delay` | `float` | `None`→auto | Auto: `3.0` s on CUDA, `0.75` s on CPU/MPS (pass a number to override) |
| `initial_value` | `int` | `None`→auto | Auto: `512` CUDA, `64` MPS, `32` CPU (pass a number to override) |
| `n_attempts` | `int` | `50` | Maximum attempts |
| `inference_only` | `bool` | `False` | If `True`, no backward. If `False`, forward + backward. |
| `factor_down` | `float` | `2.0` | After failure: `next = value / factor_down` |
| `factor_up` | `float` | `2.0` | Used when `memory_guided=False` for success steps |
| `memory_guided` | `bool` | `True` | Use peak GPU memory (and optional CPU via `psutil`) to pick the next size; cap with `max_growth_multiplier` |
| `memory_target_fraction` | `float` | `0.88` | Target peak VRAM as a fraction of total GPU memory when extrapolating |
| `max_growth_multiplier` | `float` | `6.0` | Max single-step increase after success |
| `cuda_mem_devices` | `int`, list, or `"all"` | `None` | Which GPUs to read peak memory on; default = search `device`’s index. `"all"` or `[0,1,…]` for multi-GPU bottleneck |
| `use_subprocess` | `bool` | `None` | Default: subprocess on **Linux/macOS** for CUDA, CPU, and MPS (worker may be OOM-killed; parent continues). Set ``BATCH_FINDER_SUBPROCESS=0`` for in-process on very tight hosts. Windows: in-process |

**Returns:** Shape tuple (tuple/list `input_shapes`), or `int` (string/dict `input_shapes` or `axis_to_maximize`), or `None` if nothing worked.

**Modes:**

- **Tuple/list:** first `forward` argument; `-1` marks the free axis, or add negative floats to scale other axes off that trial size.
- **String:** one shape group per tensor; names, one `=-1`, optional equations between names.
- **Dict:** like the string form, keys = parameter names, `"#constraints"` for the rules.
- **`axis_to_maximize` + `fixed_axis`:** when you skip `input_shapes`.

**Example output** (`axis_to_maximize` + `fixed_axis`):


```
--- Detected inputs (type, estimated shape) ---
  input_ids: integer, (32, 64)
  attention_mask: integer, (32, 64)
---

batch_size fixed={'seq_len': 32}: 100%|████████████████████| 22/50 [01:26<00:00,  3.9s/it, gpus=1, i=22/50, max_ok=1919, min_fail=1920, status=✅, value=1919

✅ Max value that passed: 1919
```



## 🔧 How It Works

1. **Inputs** – Uses `input_shapes` dict keys or `forward_params` (no live module in the parent during the search).
2. **Types** – Integers for `*ids`, `*mask`, `labels`; floats otherwise.
3. **Shapes** – From optional `config`, or from `get_model().config` (one probe if you omit `config`) plus argument names.
4. **Search** – On success, grow; on failure, shrink. Stops at failure at size `1` or when `n_attempts` is reached.
5. **Loss** – Uses `output.loss` if present, otherwise sums output tensors.


## ⚠️ Important Notes

- **Memory:** Lower `initial_value` on small GPUs.
- **Speed:** `inference_only=True` is faster.
- **Training:** `inference_only=False` runs backward too.
- **Size 1:** If size `1` fails, the function returns `None`.
- **OOM:** CUDA OOM and similar errors are caught and the search continues. On **Linux/macOS**, attempts default to a **subprocess** so the **parent** keeps running if the host kills the worker (SIGKILL). Set `BATCH_FINDER_SUBPROCESS=0` for in-process attempts if repeated `spawn` reloads are worse on your host. Each worker run ends with `del model`, `gc.collect()`, and CUDA/MPS cache clears where applicable. After failures, the same cleanup runs.
- **Defaults:** Omitting `initial_value` and `delay` picks smaller starts and shorter pauses on CPU/MPS than on CUDA.
- **Login nodes:** Prefer a real GPU job for meaningful limits. On CPU-only hosts, use a smaller `initial_value` and/or `inference_only=True` if RAM is tight.
- **Memory-guided steps:** After each good run, peak GPU memory vs total guides the next step (capped by `max_growth_multiplier`). Install `psutil` for a rough CPU hint. Set `memory_guided=False` to use only `factor_up` / `factor_down`.
- **Multi-GPU:** Use `cuda_mem_devices="all"` or a list so every GPU is considered; the smallest safe step applies.



## 🤝 Contributing

Contributions are welcome. Please open a Pull Request.

## 📝 License

MIT — see the LICENSE file.

---

**Made with ❤️ for the PyTorch community**

