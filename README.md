# 🔍 Batch Finder

**Find the maximum batch size, documents, and timesteps your PyTorch models can handle without running out of memory.**

Batch Finder is a powerful utility package that helps you determine the optimal batch size, number of documents, and sequence length for your PyTorch models. It uses binary search to efficiently find the maximum values that work without OOM errors.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## ✨ Features

- 🎯 **Find Maximum Batch Size** - Determine the largest batch size your model can handle
- 📚 **Find Maximum Documents** - Test multi-document inputs (e.g., for RAG models)
- ⏱️ **Find Maximum Timesteps** - Discover the longest sequence length your model supports
- 🚀 **Inference Mode** - Option to test without computing gradients (faster, less memory)
- 🔧 **Custom Functions** - Use your own forward pass logic for complex models
- 🛡️ **Safe Testing** - Automatic error handling and memory cleanup
- 📊 **Progress Tracking** - Beautiful progress bars with real-time status

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

### Basic Usage

```python
import torch
from batch_finder import batch_finder

# Your model
model = torch.nn.Transformer(...)

# Find maximum batch size, docs, and timesteps
results = batch_finder(model)

print(f"Max batch size: {results['batch_size']}")
print(f"Max documents: {results['n_docs']}")
print(f"Max timesteps: {results['timesteps']}")
```

### Find Only Batch Size

```python
from batch_finder import find_max_batch

model = MyModel()
max_batch = find_max_batch(
    model=model,
    input_shape=(128, 768),  # (seq_len, features)
    max_batch_size=64
)
print(f"Maximum batch size: {max_batch}")
```

### Find Only Documents

```python
from batch_finder import find_max_docs

model = MyRAGModel()
max_docs = find_max_docs(
    model=model,
    batch_size=4,
    seq_len=128,
    max_docs=100
)
print(f"Maximum documents: {max_docs}")
```

### Find Only Timesteps

```python
from batch_finder import find_max_timesteps

model = MyModel()
max_timesteps = find_max_timesteps(
    model=model,
    batch_size=4,
    max_timesteps=2048
)
print(f"Maximum timesteps: {max_timesteps}")
```

## 🎛️ Advanced Usage

### Inference-Only Mode

For faster testing without gradient computation:

```python
results = batch_finder(
    model=model,
    inference_only=True  # No gradients computed
)
```

### Custom Forward Function

For models with complex input requirements:

```python
def my_custom_forward(model, batch_input, **kwargs):
    # Your custom forward logic
    encoder_input = kwargs.get('encoder_input')
    return model(
        input_ids=batch_input,
        input_ids_encoder=encoder_input,
        attention_mask=kwargs.get('attention_mask')
    )

results = batch_finder(
    model=model,
    custom_forward=my_custom_forward,
    attention_mask=torch.ones(4, 128)
)
```

### Custom Ranges

```python
results = batch_finder(
    model=model,
    batch_size_range=(1, 256),      # Test batch sizes 1-256
    docs_range=(1, 500),              # Test 1-500 documents
    timesteps_range=(128, 4096),      # Test sequence lengths 128-4096
    delay=2.0                         # 2 second delay between tests
)
```

### Specific Device

```python
device = torch.device('cuda:0')
results = batch_finder(
    model=model,
    device=device
)
```

## 📖 API Reference

### `batch_finder(model, ...)`

Main convenience function to find all maximum values.

**Parameters:**
- `model` (torch.nn.Module): Your PyTorch model
- `device` (torch.device, optional): Device to run on (default: auto-detect)
- `batch_size_range` (Tuple[int, int]): (min, max) batch sizes (default: (1, 128))
- `docs_range` (Tuple[int, int]): (min, max) documents (default: (1, 200))
- `timesteps_range` (Tuple[int, int]): (min, max) sequence lengths (default: (1, 2048))
- `delay` (float): Delay between tests in seconds (default: 3.0)
- `inference_only` (bool): If True, use `torch.no_grad()` (default: False)
- `custom_forward` (Callable, optional): Custom forward function
- `**forward_kwargs`: Additional kwargs for model forward

**Returns:**
- `Dict[str, int]`: Dictionary with `batch_size`, `n_docs`, and `timesteps` keys

### `find_max_batch(model, input_shape, ...)`

Find maximum batch size.

**Parameters:**
- `model` (torch.nn.Module): Your PyTorch model
- `input_shape` (Tuple[int, ...]): Input shape excluding batch dimension, e.g., `(128, 768)`
- `device` (torch.device, optional): Device to run on
- `min_batch_size` (int): Minimum batch size (default: 1)
- `max_batch_size` (int): Maximum batch size (default: 128)
- `delay` (float): Delay between tests (default: 3.0)
- `inference_only` (bool): If True, no gradients (default: False)
- `custom_forward` (Callable, optional): Custom forward function
- `**forward_kwargs`: Additional kwargs

**Returns:**
- `int`: Maximum batch size that works

### `find_max_docs(model, batch_size, seq_len, ...)`

Find maximum number of documents.

**Parameters:**
- `model` (torch.nn.Module): Your PyTorch model
- `batch_size` (int): Batch size to use
- `seq_len` (int): Sequence length
- `device` (torch.device, optional): Device to run on
- `min_docs` (int): Minimum documents (default: 1)
- `max_docs` (int): Maximum documents (default: 200)
- `delay` (float): Delay between tests (default: 3.0)
- `inference_only` (bool): If True, no gradients (default: False)
- `custom_forward` (Callable, optional): Custom forward function
- `**forward_kwargs`: Additional kwargs

**Returns:**
- `int`: Maximum number of documents that works

### `find_max_timesteps(model, batch_size, ...)`

Find maximum sequence length (timesteps).

**Parameters:**
- `model` (torch.nn.Module): Your PyTorch model
- `batch_size` (int): Batch size to use
- `device` (torch.device, optional): Device to run on
- `min_timesteps` (int): Minimum sequence length (default: 1)
- `max_timesteps` (int): Maximum sequence length (default: 2048)
- `delay` (float): Delay between tests (default: 3.0)
- `inference_only` (bool): If True, no gradients (default: False)
- `custom_forward` (Callable, optional): Custom forward function
- `**forward_kwargs`: Additional kwargs

**Returns:**
- `int`: Maximum sequence length that works

## 💡 Examples

### Example 1: Transformer Model

```python
from transformers import AutoModelForCausalLM
from batch_finder import batch_finder

model = AutoModelForCausalLM.from_pretrained("gpt2")
results = batch_finder(model, inference_only=True)
```

### Example 2: RAG Model with Custom Forward

```python
from batch_finder import find_max_docs

def rag_forward(model, docs_input, batch_input, **kwargs):
    # docs_input: (n_docs, batch_size, seq_len)
    # batch_input: (batch_size, seq_len)
    return model(
        input_ids=batch_input,
        input_ids_encoder=docs_input
    )

max_docs = find_max_docs(
    model=rag_model,
    batch_size=4,
    seq_len=128,
    custom_forward=rag_forward,
    max_docs=100
)
```

### Example 3: Testing Before Training

```python
from batch_finder import find_max_batch

# Find max batch size for training
max_batch = find_max_batch(
    model=model,
    input_shape=(512,),  # sequence length
    max_batch_size=64,
    inference_only=False  # Need gradients for training
)

print(f"Use batch_size={max_batch} for training")
```

## 🔧 How It Works

Batch Finder uses **binary search** to efficiently find the maximum values:

1. **Binary Search**: Tests values in a binary search pattern (logarithmic time complexity)
2. **Safe Testing**: Each test runs in a try-except block with automatic cleanup
3. **Memory Management**: Clears CUDA cache after each test
4. **Progress Tracking**: Shows real-time progress with tqdm progress bars

## ⚠️ Important Notes

- **Memory**: Make sure you have enough GPU/CPU memory. Start with conservative max values.
- **Time**: Testing can take time, especially with large models. Use `inference_only=True` for faster results.
- **Gradients**: Set `inference_only=False` if you need to test with gradients (e.g., for training).
- **Custom Models**: For complex models, use `custom_forward` to define your own forward pass.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

This package was inspired by the need to efficiently find optimal batch sizes for large language models and RAG systems.

## 📧 Contact

For questions, issues, or suggestions, please open an issue on GitHub.

---

**Made with ❤️ for the PyTorch community**

