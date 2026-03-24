"""
Batch Finder - Find maximum batch size, documents, and timesteps for your PyTorch models.

A utility package to help you determine the maximum batch size, number of documents,
and sequence length (timesteps) that your model can handle without running out of memory.
"""

from .batch_finder import find_max_minibatch
from .input_shapes import (
    CONSTRAINTS_KEY,
    FINDER_CONSTRAINTS_KEY,
    InputShapesSpec,
    materialize_shapes,
    parse_input_shapes,
    parse_input_shapes_dict,
)

__version__ = "0.1.0"
__all__ = [
    "find_max_minibatch",
    "parse_input_shapes",
    "parse_input_shapes_dict",
    "CONSTRAINTS_KEY",
    "FINDER_CONSTRAINTS_KEY",
    "InputShapesSpec",
    "materialize_shapes",
]

