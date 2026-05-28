"""Readable building blocks for the sparse switching mean-variance strategy."""

from .math_utils import _condition_number, _is_finite_pos, _project_to_capped_simplex, _safe_float

__all__ = [
    "_condition_number",
    "_is_finite_pos",
    "_project_to_capped_simplex",
    "_safe_float",
]
