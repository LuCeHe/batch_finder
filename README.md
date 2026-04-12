<h1 align="center">🔍 Batch Finder</h1>

<p align="center"><strong>Find the maximum value for any dimension your PyTorch models can handle without stopping the code.</strong></p>

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.8+-blue.svg" alt="Python 3.8+"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-1.9+-orange.svg" alt="PyTorch"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
</p>

Batch Finder detects your model’s inputs (types and shapes), fixes the sizes you choose, and searches for the largest value your run can sustain without stopping the code.



## ✨ Features

- 🎯 **One main function** – You call `find_max_minibatch` for pretty much everything you need here.
- 🔍 **Tell it what goes in** – You pass names, data types, and rough shapes (`input_shapes` / `forward_params`). If your model has a `config`, that can help too.
- 📐 **Shapes your way** – Use a simple tuple or list, or a **list of tuples** when you have multiple tensors (for example `[(-1, 128, 512), (-1, 128, 512)]`). For several inputs, you can use a text string or a dict. Or skip that and use `axis_to_maximize` plus `fixed_axis` (handy for Hugging Face–style models).
- 🚀 **Runs forward or full training** – Turn backward on or off depending on what you want to test.
- ⚙️ **Knobs you can turn** – Change how fast it steps up or down (`factor_up`, `factor_down`), how many tries it gets (`n_attempts`), and where it starts (`initial_value`).
- ⏱️ **Time cap (optional)** – Set `time_limit_seconds` if you only want the search to run for so long. When time is up, you get the **best size that worked so far** (or `None` if nothing worked yet). Leave it unset to ignore the clock and rely on `n_attempts` and the normal stop rules.
- 🛡️ **Fails without trashing your session** – When things blow up, it cleans up. If even batch size `1` fails, you get `None` (honest “no”).
- 📊 **See what it’s doing** – `tqdm` shows a progress bar with useful status text.

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

Other **negative integers** `d < -1` (e.g. `-2`, `-3`) size that dimension as `round(|d| × trial)`—the same scaling role as negative floats below (e.g. `-2` → 2× the trial size tied to `-1`).

If the tuple mixes integers and **negative floats**, you are in **compact numeric** mode: there must be **at least one** integer `-1` (the searched axis). Any other dimension given as a **negative float** `-x` is sized as `round(|x| × trial)`, where `trial` is the current value on the `-1` axis—so `|x|` is the proportion you want between that axis and the searched axis (e.g. `-1.5` keeps that dim about 1.5× the trial size).

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

### Several tensors: list of shape tuples

Pass a **list or tuple of per-tensor shapes** (same order as `forward` / `forward_params`). Each entry follows the same rules as a single-tensor tuple (integer `-1`, integer multipliers `-2`, `-3`, …, or compact floats). **One** trial size is searched; every `-1` and every `d < -1` uses that trial value as above.

```python
# Example: two tensors, same free shape pattern
max_val = find_max_minibatch(
    get_model,
    input_shapes=[(-1, 128, 512), (-1, 128, 512)],
    forward_params=["x", "y"],
)
```

For symbolic names and constraints across tensors, use a string or dict below.

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

When `forward` takes **multiple tensors**, pass **`input_shapes` as text**: one `(…)` group per argument (same order as `forward`), short names for axes that must match, optional equations between names, and **at least one** `name=-1` for the size you search.



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
    time_limit_seconds=120.0,  # optional: return best trial so far before timeout
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
| `input_shapes` | `str` \| `dict` \| tuple/list | `None` | **String:** several tensors, shared names and rules. **Dict:** named shapes + `"#constraints"`. **Flat tuple/list:** first tensor only (ints with `-1` or compact floats). **Nested list/tuple of shapes:** one tuple per tensor, e.g. `[(-1, 128, 512), (-1, 128, 512)]`. Not used together with `axis_to_maximize` |
| `axis_to_maximize` | `str` | `None` | Name of the axis when `input_shapes` is omitted, e.g. `"batch_size"` |
| `fixed_axis` | `Dict[str, int]` | `{}` | Fixed sizes, e.g. `{"seq_len": 128}` |
| `device` | `torch.device` | auto | Device to run on |
| `delay` | `float` | `None`→auto | Auto: `3.0` s on CUDA, `0.75` s on CPU/MPS (pass a number to override) |
| `initial_value` | `int` | `None`→auto | Auto: `512` CUDA, `64` MPS, `32` CPU (pass a number to override) |
| `n_attempts` | `int` | `50` | Maximum attempts |
| `time_limit_seconds` | `float` | `None` | Wall-clock limit **from the start of the search loop** (after the probe). When time is up, returns the **largest successful trial** so far (or `None` if none). Shortens sleeps between attempts; with subprocess workers, **terminates** a run that would exceed the remaining time. In-process forward/backward is not interrupted mid-step. |
| `inference_only` | `bool` | `False` | If `True`, no backward. If `False`, forward + backward. |
| `factor_down` | `float` | `2.0` | After failure: `next = value / factor_down` |
| `factor_up` | `float` | `2.0` | Used when `memory_guided=False` for success steps |
| `memory_guided` | `bool` | `True` | Use peak GPU memory (and optional CPU via `psutil`) to pick the next size; cap with `max_growth_multiplier` |
| `memory_target_fraction` | `float` | `0.88` | Target peak VRAM as a fraction of total GPU memory when extrapolating |
| `max_growth_multiplier` | `float` | `6.0` | Max single-step increase after success |
| `cuda_mem_devices` | `int`, list, or `"all"` | `None` | Which GPUs to read peak memory on; default = search `device`’s index. `"all"` or `[0,1,…]` for multi-GPU bottleneck |
| `use_subprocess` | `bool` | `None` | Default: subprocess on **Linux/macOS** for CUDA, CPU, and MPS (worker may be OOM-killed; parent continues). Set ``BATCH_FINDER_SUBPROCESS=0`` for in-process on very tight hosts. Windows: in-process |

**Returns:** Shape tuple, **tuple of shape tuples** (multi-tensor list `input_shapes`), or `int` (string/dict `input_shapes` or `axis_to_maximize`), or `None` if nothing worked. With **`time_limit_seconds`**, you may get the best value found before the deadline even if `n_attempts` was not reached.

**Modes:**

- **Tuple/list:** first `forward` argument, or a **list/tuple of shape tuples** for several arguments; `-1` marks the free axis, or add negative floats to scale other axes off that trial size.
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
4. **Search** – On success, grow; on failure, shrink. Stops at failure at size `1`, when `n_attempts` is reached, or when **`time_limit_seconds`** elapses (then returns the best successful trial so far, if any).
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
- **Multi-GPU (one OS process):** Use `cuda_mem_devices="all"` or a list so every visible GPU is measured; memory-guided growth uses the tightest GPU (minimum headroom).
- **DDP / torchrun / Accelerate (several processes):** Each rank runs the search on **its** assigned GPU (`cuda_mem_devices` default is that GPU only). When ``WORLD_SIZE > 1``, ``find_max_minibatch`` **by default** takes the **minimum** successful trial size across ranks (JSON file sync) before returning, so every rank agrees on the same limit. Pass ``distributed_sync_dir=…`` to a directory on **shared** storage (or set ``BATCH_FINDER_SYNC_DIR``); otherwise sync files go under ``$WORK/.cache`` if ``WORK`` is set, else ``$HOME/.cache``.
- **Time limit:** Pass `time_limit_seconds=…` to cap wall-clock time for the **search loop** (after the one-time probe). You get the best batch found so far when the limit hits. Subprocess workers are stopped if they would run past the remaining budget; an in-process forward/backward still runs to completion once started.



## 🤝 Contributing

Contributions are welcome. Please open a Pull Request.

## 📝 License

MIT — see the LICENSE file.

---

**Made with ❤️ for the PyTorch community**

