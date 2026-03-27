"""
Example usage of batch_finder package.
"""

import sys
from pathlib import Path

# Prefer this repo when running `python example.py` (avoids an older `batch_finder` on PYTHONPATH).
sys.path.insert(0, str(Path(__file__).resolve().parent))

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


# Example 3: Two inputs — x (23,b,t,45), y (b,t,12); shared axis b (see input_shapes DSL)
# Sum over non-b dimensions on each tensor → two vectors of length b, add, L2 norm.
class TwoInputSummedNormModel(nn.Module):
    def forward(self, x, y):
        # x: (23, b, t, 45) — b is dim 1; sum over 23, t, 45
        v1 = x.sum(dim=(0, 2, 3))
        # y: (b, t, 12) — b is dim 0; sum over t, 12
        v2 = y.sum(dim=(1, 2))
        return torch.norm(v1 + v2)


class TwoInputSummedNormModelWithParam(TwoInputSummedNormModel):
    """Same forward path, plus a scalar nn.Parameter so backward hits weights and inputs."""

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones((), dtype=torch.float32))

    def forward(self, x, y):
        v1 = x.sum(dim=(0, 2, 3))
        v2 = y.sum(dim=(1, 2))
        return self.scale * torch.norm(v1 + v2)


def get_simple_model() -> SimpleModel:
    return SimpleModel()


def get_two_input_model() -> TwoInputSummedNormModel:
    return TwoInputSummedNormModel()


def get_two_input_model_with_param() -> TwoInputSummedNormModelWithParam:
    return TwoInputSummedNormModelWithParam()


# Two-input DSL examples: tensor argument names in forward order (same as string DSL groups).
# TwoInputForwardParams = ("x", "y")


def get_distilgpt2():
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained("distilgpt2")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch Finder Examples")
    parser.add_argument("--n-attempts", "-n", type=int, default=2, help="Max attempts per test (default: 50)")
    args = parser.parse_args()

    experiments_to_run = [
        'simple',
        'hf',
        'two_input',
    ]

    print("=" * 60)
    print("Batch Finder Examples")
    print("=" * 60)

    n_attempts = args.n_attempts
    n_attempts_non_hf = 2

    if 'simple' in experiments_to_run:
        # 1a: (a, b, -1) - fixed a,b, maximize last axis
        print("\n1a. SimpleModel input_shapes=(4, 8, -1)...")
        res_1a = find_max_minibatch(
            get_simple_model,
            input_shapes=(4, 8, -1),
            n_attempts=n_attempts_non_hf,
        )
        print(f"   Result: Final input shape = {res_1a}")

        # 1b: (-1, b, c, d) - maximize axis 0
        print("\n1b. SimpleModel input_shapes=(-1, 4, 8, 16)...")
        res_1b = find_max_minibatch(
            get_simple_model,
            input_shapes=(-1, 4, 8, 16),
            n_attempts=n_attempts_non_hf,
        )
        print(f"   Result: Final input shape = {res_1b}")

        # 1c: compact — float -1. as search axis, -1.3 scales another dim vs. trial
        print("\n1c. SimpleModel input_shapes=(-1., 4, -1.3, 16)...")
        res_1c = find_max_minibatch(
            get_simple_model,
            input_shapes=(-1., 4, -1.3, 16),
            n_attempts=n_attempts_non_hf,
        )
        print(f"   Result: Final input shape = {res_1c}")

    if 'hf' in experiments_to_run:

        # Example 2: Small HuggingFace model (requires: pip install transformers)
        try:
            import transformers  # noqa: F401 — dependency check only

            print("\n2. HuggingFace model (distilgpt2)...")
            max_batch_hf = find_max_minibatch(
                get_distilgpt2,
                axis_to_maximize="batch_size",
                fixed_axis={"seq_len": 32},
                n_attempts=n_attempts,
            )
            print(f"   Result: Max batch_size = {max_batch_hf}")

            # 2b: Fix batch_size, maximize seq_len
            print("\n2b. HuggingFace model – fix batch_size=2, maximize seq_len...")
            max_seq_hf = find_max_minibatch(
                get_distilgpt2,
                axis_to_maximize="seq_len",
                fixed_axis={"batch_size": 2},
                n_attempts=n_attempts,
            )
            print(f"   Result: Max seq_len = {max_seq_hf}")
        except ImportError:
            print("\n2. Skipping HF example (pip install transformers to enable)")

    if 'two_input' in experiments_to_run:
        dsl = "(23, b, t, 45),(b, t, 12), t=1.5b, b=-1"

        # 3: Two-input model (no parameters) — t=1.5b (rounded), search maximizes b
        print(f"\n3. TwoInputSummedNormModel — input_shapes '{dsl}'...")
        res_3 = find_max_minibatch(
            get_two_input_model,
            input_shapes=dsl,
            # inference_only=False,
            n_attempts=n_attempts_non_hf,
            # forward_params=TwoInputForwardParams,
        )
        print(f"   Result: max b = {res_3}")

        # 4: Same architecture + scalar nn.Parameter (scale * ||v1+v2||)
        print(f"\n4. TwoInputSummedNormModelWithParam — same input_shapes '{dsl}'...")
        res_4 = find_max_minibatch(
            get_two_input_model_with_param,
            input_shapes=dsl,
            # inference_only=False,
            n_attempts=n_attempts_non_hf,
            # forward_params=TwoInputForwardParams,
        )
        print(f"   Result: max b = {res_4}")

    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)
