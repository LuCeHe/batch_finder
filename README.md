# 🔍 Batch Finder

**Find the maximum value for any dimension your PyTorch models can handle without running out of memory.**

Batch Finder automatically detects your model's inputs (type and shape), fixes the dimensions you specify, and finds the maximum value for the remaining axis using a configurable search strategy.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## ✨ Features

- 🎯 **Single unified API** – One function `find_max_minibatch` for all cases
- 🔍 **Automatic input detection** – Infers input names, types (int/float), and shapes from the model
- 📐 **Flexible shape specification** – Use `input_shape` with `-1` for the variable axis, or `axis_to_maximize` + `fixed_axis`
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

### Mode 1: `input_shape` (single-input models)

Use a tuple with `-1` for the axis to maximize and numbers for fixed dimensions:

```python
from batch_finder import find_max_minibatch

model = MyModel()

# Maximize axis 0, fix (64, 256)
max_val = find_max_minibatch(
    model=model,
    input_shape=(-1, 64, 256),
    initial_value=64,
)

# Maximize axis 2, fix (4, 8)
max_val = find_max_minibatch(model=model, input_shape=(4, 8, -1), initial_value=32)

# Multiple -1: same value for all variable axes
max_val = find_max_minibatch(model=model, input_shape=(-1, 4, -1, 16), initial_value=32)
```

### Mode 2: `axis_to_maximize` + `fixed_axis` (multi-input models, e.g. HuggingFace)

```python
from transformers import AutoModelForCausalLM
from batch_finder import find_max_minibatch

model = AutoModelForCausalLM.from_pretrained("distilgpt2")
max_batch = find_max_minibatch(
    model=model,
    axis_to_maximize="batch_size",
    fixed_axis={"seq_len": 32},
    initial_value=32,
)
print(f"Max batch size: {max_batch}")
```

### Custom search parameters

```python
max_val = find_max_minibatch(
    model=model,
    input_shape=(-1, 128, 512),
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
| `input_shape` | `Tuple[int, ...]` | `None` | Shape with `-1` for variable axis(s), e.g. `(-1, 64, 256)` |
| `axis_to_maximize` | `str` | `None` | Axis name when not using `input_shape`, e.g. `"batch_size"` |
| `fixed_axis` | `Dict[str, int]` | `{}` | Fixed values, e.g. `{"seq_len": 128}` |
| `device` | `torch.device` | auto | Device to run on |
| `delay` | `float` | `3.0` | Seconds between attempts |
| `initial_value` | `int` | `1024` | First value to try |
| `n_attempts` | `int` | `50` | Maximum attempts |
| `inference_only` | `bool` | `False` | If `True`, no gradients. If `False`, full forward+backward. |
| `factor_down` | `float` | `2.0` | On failure: `next = value / factor_down` |
| `factor_up` | `float` | `2.0` | On success: `next = value * factor_up` |

**Returns:** `Tuple[int, ...]` (when using `input_shape`), `int` (when using `axis_to_maximize`), or `None` if none passed.

**Modes:**
- Provide `input_shape`: uses first input param with the given shape; `-1` = variable axis.
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

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Made with ❤️ for the PyTorch community**
