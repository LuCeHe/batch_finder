"""
Example usage of batch_finder package.
"""

import argparse
import torch
import torch.nn as nn
from batch_finder import find_max_minibatch


# Example 1: Small PyTorch model
# Accepts (a,b), (a,b,c), (a,b,c,d) - flexible last dim for -1 in any position
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.activation = nn.ReLU()
        self._linears = nn.ModuleDict()

    def _get_linear(self, in_features: int, device):
        key = str(in_features)
        if key not in self._linears:
            self._linears[key] = nn.Linear(in_features, in_features).to(device)
        return self._linears[key]

    def forward(self, x):
        in_features = x.shape[-1]
        linear = self._get_linear(in_features, x.device)
        return self.activation(linear(x))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch Finder Examples")
    parser.add_argument("--n-attempts", "-n", type=int, default=2, help="Max attempts per test (default: 50)")
    args = parser.parse_args()

    print("=" * 60)
    print("Batch Finder Examples")
    print("=" * 60)

    n_attempts = args.n_attempts
    model1 = SimpleModel()

    # 1a: (a, b, -1) - fixed a,b, maximize last axis
    print("\n1a. SimpleModel input_shape=(4, 8, -1)...")
    max_1a = find_max_minibatch(model=model1, input_shape=(4, 8, -1), initial_value=64, inference_only=False, n_attempts=n_attempts)
    print(f"   Result: Max = {max_1a}")

    # 1b: (-1, b, c, d) - maximize axis 0
    print("\n1b. SimpleModel input_shape=(-1, 4, 8, 16)...")
    max_1b = find_max_minibatch(model=model1, input_shape=(-1, 4, 8, 16), initial_value=64, inference_only=True, n_attempts=n_attempts)
    print(f"   Result: Max = {max_1b}")

    # 1c: (-1, b, -1, d) - maximize axes 0 and 2 (same value)
    print("\n1c. SimpleModel input_shape=(-1, 4, -1, 16)...")
    max_1c = find_max_minibatch(model=model1, input_shape=(-1, 4, -1, 16), initial_value=32, inference_only=True, n_attempts=n_attempts)
    print(f"   Result: Max = {max_1c}")

    # Example 2: Small HuggingFace model (requires: pip install transformers)
    try:
        from transformers import AutoModelForCausalLM

        print("\n2. HuggingFace model (distilgpt2)...")
        model2 = AutoModelForCausalLM.from_pretrained("distilgpt2")
        max_batch_hf = find_max_minibatch(
            model=model2,
            axis_to_maximize="batch_size",
            fixed_dims={"seq_len": 32},
            initial_value=32,
            inference_only=False,
            n_attempts=n_attempts,
        )
        print(f"   Result: Max batch_size = {max_batch_hf}")
    except ImportError:
        print("\n2. Skipping HF example (pip install transformers to enable)")

    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)
