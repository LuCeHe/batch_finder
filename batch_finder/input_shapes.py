"""
Parse and evaluate the input_shapes DSL for multi-tensor symbolic dimensions.

Example:
    '(23, b, t, 45),(b, t, 12), t=1.5b, b=-1'

- Tuple groups are one shape per forward tensor argument (same order as forward).
- Dimensions are non-negative integers or identifiers (shared axes use the same name).
- After the tuples: comma-separated assignments. Exactly one must be ``symbol=-1`` (the
  dimension to maximize). Others define derived sizes, e.g. ``t=1.5b`` (rounded to int).
"""

from __future__ import annotations

import numbers
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

Dim = Union[int, str]

# Reserved key for dict-style ``input_shapes`` (see ``parse_input_shapes_dict``).
CONSTRAINTS_KEY = "#constraints"
FINDER_CONSTRAINTS_KEY = CONSTRAINTS_KEY  # backward-compatible alias


@dataclass(frozen=True)
class InputShapesSpec:
    """Parsed ``input_shapes`` string."""

    shapes: List[List[Dim]]
    search_symbol: str
    constraints: Tuple[Tuple[str, str], ...]


def _parse_dim_token(tok: str) -> Dim:
    t = tok.strip()
    if not t:
        raise ValueError("empty dimension in tuple")
    if re.fullmatch(r"-?\d+", t):
        v = int(t)
        if v < 0:
            raise ValueError(
                f"negative dimension {v!r} in tuple is not allowed; use a named symbol and symbol=-1"
            )
        return v
    if re.fullmatch(r"[a-zA-Z_][a-zA-Z0-9_]*", t):
        return t
    raise ValueError(f"invalid dimension token: {tok!r}")


def _extract_tuple_groups(s: str) -> Tuple[List[str], str]:
    """Return list of inner strings for each ``(...)`` group and the remainder."""
    s = s.strip()
    groups: List[str] = []
    i = 0
    n = len(s)
    while i < n:
        while i < n and s[i] in " \t,":
            i += 1
        if i >= n:
            break
        if s[i] != "(":
            break
        depth = 0
        start = i
        while i < n:
            if s[i] == "(":
                depth += 1
            elif s[i] == ")":
                depth -= 1
                if depth == 0:
                    inner = s[start + 1 : i]
                    groups.append(inner)
                    i += 1
                    break
            i += 1
        else:
            raise ValueError("unclosed '(' in input_shapes")
    rest = s[i:].strip()
    return groups, rest


def _split_constraints(rest: str) -> Tuple[str, List[Tuple[str, str]]]:
    if not rest:
        raise ValueError("expected constraints after shape tuples (e.g. b=-1)")
    parts = [p.strip() for p in rest.split(",") if p.strip()]
    search_symbol: Optional[str] = None
    constraints: List[Tuple[str, str]] = []
    for p in parts:
        if "=" not in p:
            raise ValueError(f"expected name=value, got: {p!r}")
        lhs, rhs = p.split("=", 1)
        lhs, rhs = lhs.strip(), rhs.strip()
        if rhs == "-1":
            if search_symbol is not None:
                raise ValueError("only one symbol=-1 (search dimension) is allowed")
            search_symbol = lhs
        else:
            constraints.append((lhs, rhs))
    if not search_symbol:
        raise ValueError("exactly one assignment must be symbol=-1 (the dimension to maximize)")
    return search_symbol, constraints


def _parse_shape_entry_str(s: str) -> Tuple[List[Dim], Optional[str]]:
    """
    Parse one forward-argument entry like ``(b, t)`` or ``(b, t), int``.

    Returns (dims, dtype_hint) where dtype_hint is ``"integer"``, ``"float"``, or ``None``.
    """
    s = s.strip()
    if not s.startswith("("):
        raise ValueError(f"shape entry must start with '(': {s!r}")
    depth = 0
    close_idx: Optional[int] = None
    for i, c in enumerate(s):
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                close_idx = i
                break
    if close_idx is None:
        raise ValueError(f"unclosed '(' in {s!r}")
    inner = s[1:close_idx]
    rest = s[close_idx + 1 :].strip()
    dtype_hint: Optional[str] = None
    if rest:
        if not rest.startswith(","):
            raise ValueError(f"unexpected trailing content after shape tuple in {s!r}")
        hint = rest[1:].strip().lower()
        if hint in ("int", "integer"):
            dtype_hint = "integer"
        elif hint in ("float",):
            dtype_hint = "float"
        elif hint:
            raise ValueError(f"unknown dtype hint {hint!r} in {s!r}; use int or float")
    toks = [t.strip() for t in inner.split(",") if t.strip()]
    if not toks:
        raise ValueError(f"empty shape tuple in {s!r}")
    dims = [_parse_dim_token(t) for t in toks]
    return dims, dtype_hint


def parse_input_shapes_dict(
    mapping: Mapping[str, Any],
    forward_param_order: List[str],
) -> Tuple[InputShapesSpec, Dict[str, str]]:
    """
    Build an :class:`InputShapesSpec` from a dict keyed by ``forward`` parameter names.

    Each tensor argument maps to a string ``\"(dims...), optional_dtype\"``, e.g.
    ``\"(b, t), int\"`` or ``\"(d, b, t)\"``. Optional dtype after the tuple must be
    ``int`` or ``float`` (default is inferred from the parameter name).

    The special key :data:`CONSTRAINTS_KEY` (``\"#constraints\"``) holds the
    same constraint list as the string DSL after the shape tuples, e.g.
    ``\"t=2b, b=-1\"`` — exactly one assignment must be ``symbol=-1`` (search dimension).

    Args:
        mapping: Dict including ``#constraints`` and one entry per name in
            ``forward_param_order``.
        forward_param_order: Ordered ``forward`` tensor parameter names (e.g. from the model).
    """
    if CONSTRAINTS_KEY not in mapping:
        raise ValueError(
            f"input_shapes dict must include {CONSTRAINTS_KEY!r} "
            '(e.g. \"t=2b, b=-1\").'
        )
    fc = mapping[CONSTRAINTS_KEY]
    if not isinstance(fc, str):
        raise TypeError(f"{CONSTRAINTS_KEY!r} must be a string.")
    allowed = set(forward_param_order) | {CONSTRAINTS_KEY}
    unknown = set(mapping.keys()) - allowed
    if unknown:
        raise ValueError(f"Unknown keys in input_shapes dict: {sorted(unknown)}")
    missing = set(forward_param_order) - set(mapping.keys())
    if missing:
        raise ValueError(
            f"input_shapes dict missing keys for forward params: {sorted(missing)}"
        )

    shapes: List[List[Dim]] = []
    dtype_overrides: Dict[str, str] = {}
    for name in forward_param_order:
        raw = mapping[name]
        if not isinstance(raw, str):
            raise TypeError(f"input_shapes[{name!r}] must be a string.")
        dims, hint = _parse_shape_entry_str(raw)
        shapes.append(dims)
        if hint is not None:
            dtype_overrides[name] = hint

    search_symbol, constraints = _split_constraints(fc.strip())
    spec = InputShapesSpec(
        shapes=shapes,
        search_symbol=search_symbol,
        constraints=tuple(constraints),
    )
    return spec, dtype_overrides


def parse_input_shapes(s: str) -> InputShapesSpec:
    """
    Parse an input_shapes string into shapes, search symbol, and derived constraints.

    Shape tuples use commas; after the last ``)`` list constraints separated by commas.
    """
    groups, rest = _extract_tuple_groups(s)
    if not groups:
        raise ValueError("expected at least one '(...)' shape tuple")
    shapes: List[List[Dim]] = []
    for inner in groups:
        toks = [t.strip() for t in inner.split(",") if t.strip()]
        if not toks:
            raise ValueError("empty shape tuple")
        shapes.append([_parse_dim_token(t) for t in toks])
    search_symbol, constraints = _split_constraints(rest)
    return InputShapesSpec(
        shapes=shapes,
        search_symbol=search_symbol,
        constraints=tuple(constraints),
    )


def _eval_rhs(rhs: str, bindings: Dict[str, int]) -> int:
    rhs = rhs.strip()
    if re.fullmatch(r"-?\d+", rhs):
        return int(rhs)
    m = re.fullmatch(r"([+-]?\d*\.?\d+)\s*\*\s*([a-zA-Z_][a-zA-Z0-9_]*)", rhs)
    if m:
        coef = float(m.group(1))
        name = m.group(2)
        if name not in bindings:
            raise KeyError(f"unknown symbol {name!r} in {rhs!r}")
        return int(round(coef * bindings[name]))
    m = re.fullmatch(r"([+-]?\d*\.?\d+)([a-zA-Z_][a-zA-Z0-9_]*)", rhs)
    if m:
        coef = float(m.group(1))
        name = m.group(2)
        if name not in bindings:
            raise KeyError(f"unknown symbol {name!r} in {rhs!r}")
        return int(round(coef * bindings[name]))
    m = re.fullmatch(r"([a-zA-Z_][a-zA-Z0-9_]*)\s*\*\s*([a-zA-Z_][a-zA-Z0-9_]*)", rhs)
    if m:
        a, b = m.group(1), m.group(2)
        if a not in bindings or b not in bindings:
            raise KeyError(f"unknown symbol in {rhs!r}")
        return bindings[a] * bindings[b]
    m = re.fullmatch(r"[a-zA-Z_][a-zA-Z0-9_]*", rhs)
    if m:
        name = m.group(0)
        if name not in bindings:
            raise KeyError(f"unknown symbol {name!r}")
        return bindings[name]
    raise ValueError(f"cannot evaluate constraint right-hand side: {rhs!r}")


def _apply_constraints(constraints: Tuple[Tuple[str, str], ...], bindings: Dict[str, int]) -> Dict[str, int]:
    out = dict(bindings)
    n = len(constraints) + 2
    for _ in range(n):
        for lhs, rhs in constraints:
            lhs = lhs.strip()
            out[lhs] = _eval_rhs(rhs, out)
    return out


def materialize_shapes(
    spec: InputShapesSpec,
    search_value: int,
    fixed_axis: Dict[str, int],
) -> Tuple[List[Tuple[int, ...]], Dict[str, int]]:
    """
    Resolve every dimension to integers. ``search_value`` is bound to ``spec.search_symbol``.
    ``fixed_axis`` supplies extra symbol bindings (literals for symbols not in tuples).
    """
    bindings: Dict[str, int] = dict(fixed_axis)
    bindings[spec.search_symbol] = search_value
    bindings = _apply_constraints(spec.constraints, bindings)

    result: List[Tuple[int, ...]] = []
    for shape in spec.shapes:
        dims: List[int] = []
        for d in shape:
            if isinstance(d, int):
                dims.append(d)
            else:
                if d not in bindings:
                    raise ValueError(f"unbound symbol {d!r} (add to fixed_axis or constraints)")
                dims.append(bindings[d])
        result.append(tuple(dims))
    return result, bindings


def collect_symbols_in_shapes(spec: InputShapesSpec) -> List[str]:
    seen: List[str] = []
    for shape in spec.shapes:
        for d in shape:
            if isinstance(d, str) and d not in seen:
                seen.append(d)
    return seen


def _coerce_compact_dim(x: Any, index: int) -> Union[int, float]:
    if isinstance(x, bool):
        raise TypeError(f"input_shapes[{index}]: boolean is not a valid dimension")
    if isinstance(x, numbers.Integral):
        return int(x)
    if isinstance(x, numbers.Real):
        return float(x)
    raise TypeError(f"input_shapes[{index}]: expected int or float, got {type(x).__name__}")


def normalize_compact_numeric_tuple(spec: Tuple[Any, ...]) -> Tuple[Union[int, float], ...]:
    """Normalize tuple/list elements to ``int`` or ``float`` (rejects bool)."""
    return tuple(_coerce_compact_dim(x, i) for i, x in enumerate(spec))


def _compact_dim_is_search_axis(d: Any) -> bool:
    """``True`` for ``-1`` (int) or ``-1.0`` / ``-1.`` (float): the binary-searched axis."""
    if isinstance(d, bool):
        return False
    return isinstance(d, numbers.Real) and d == -1


def materialize_compact_numeric_shape(
    spec: Tuple[Union[int, float], ...],
    search_value: int,
) -> Tuple[int, ...]:
    """
    Single-tensor **compact numeric** ``input_shapes`` tuple.

    - ``-1`` (int) or ``-1.0`` / ``-1.`` (float): the binary-searched axis (exactly one).
    - ``int >= 0``: fixed size.
    - ``int < -1``: invalid (use string DSL for multi-symbol search).
    - other negative ``float`` ``d``: size ``round(abs(d) * search_value)`` (e.g. ``-1.5`` → ``1.5 * s``).
    """
    n_search = sum(1 for d in spec if _compact_dim_is_search_axis(d))
    if n_search != 1:
        raise ValueError(
            "compact numeric input_shapes must contain exactly one searched axis: int -1 or "
            "float -1.0 (-1.); other negative floats scale relative to that trial size."
        )
    out: List[int] = []
    for d in spec:
        if isinstance(d, bool):
            raise TypeError(f"dimension must be int or float, got {type(d)}")
        if _compact_dim_is_search_axis(d):
            out.append(search_value)
        elif isinstance(d, int):
            if d >= 0:
                out.append(d)
            else:
                raise ValueError(
                    f"invalid int dimension {d!r}; use non-negative int, exactly one -1, "
                    "or negative float for scaled size (e.g. -1.5)"
                )
        elif isinstance(d, float):
            if d < 0:
                out.append(int(round(abs(d) * search_value)))
            else:
                raise ValueError(
                    f"non-negative float {d!r} is not allowed; use int for fixed sizes"
                )
        else:
            raise TypeError(f"dimension must be int or float, got {type(d)}")
    return tuple(out)
