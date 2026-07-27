"""Feature trees: representation, leak-safe replay, and genetic operators.

A FeatureTree is a small typed formula tree.  Leaves are dataset columns
(numeric or categorical); internal nodes are operators from the vocabulary.

Replay contract (leak-safety):
- fit_values(X_train) computes values on the training data and stores any
  fitted state (frequency maps, groupby aggregates, cross codes) on the
  nodes.  State is a function of X only, never of y.
- values(X) replays the stored state on new data without refitting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .operators import Operator, select_operators
from .types import ColumnTypes

_BIN_SYMBOL = {"add": "+", "sub": "-", "mul": "*", "div": "/"}
_GROUPBY_AGG = {
    "groupby_mean": "mean",
    "groupby_std": "std",
    "groupby_min": "min",
    "groupby_max": "max",
}


class FeatureTree:
    """One engineered feature: either a leaf column or an operator node."""

    __slots__ = ("op", "children", "column", "ctype", "state")

    def __init__(self, op=None, children=(), column=None, ctype=None, state=None):
        self.op: str | None = op
        self.children: list[FeatureTree] = list(children)
        self.column: str | None = column
        self.ctype: str | None = ctype  # leaf output type: "num" | "cat"
        self.state: dict | None = state

    # ---------------------------------------------------------------- basics
    @property
    def is_leaf(self) -> bool:
        return self.op is None

    def out_type(self, ops: dict[str, Operator]) -> str:
        if self.is_leaf:
            return self.ctype  # type: ignore[return-value]
        return ops[self.op].out_type

    def depth(self) -> int:
        if self.is_leaf:
            return 0
        return 1 + max(c.depth() for c in self.children)

    def n_nodes(self) -> int:
        if self.is_leaf:
            return 1
        return 1 + sum(c.n_nodes() for c in self.children)

    def columns_used(self) -> set[str]:
        if self.is_leaf:
            return {self.column}  # type: ignore[arg-type]
        out: set[str] = set()
        for c in self.children:
            out |= c.columns_used()
        return out

    def copy(self) -> "FeatureTree":
        if self.is_leaf:
            return FeatureTree(column=self.column, ctype=self.ctype)
        return FeatureTree(
            op=self.op,
            children=[c.copy() for c in self.children],
            state=dict(self.state) if self.state is not None else None,
        )

    # ---------------------------------------------------------------- values
    def fit_values(self, X: pd.DataFrame, ops: dict[str, Operator]) -> np.ndarray:
        return self._compute(X, ops, fit=True)

    def values(self, X: pd.DataFrame, ops: dict[str, Operator]) -> np.ndarray:
        return self._compute(X, ops, fit=False)

    def _compute(self, X: pd.DataFrame, ops: dict[str, Operator], fit: bool) -> np.ndarray:
        if self.is_leaf:
            col = X[self.column]
            if self.ctype == "num":
                return pd.to_numeric(col, errors="coerce").to_numpy(dtype=float)
            return col.to_numpy()
        op = ops[self.op]
        child_vals = [c._compute(X, ops, fit) for c in self.children]
        if not op.stateful:
            return np.asarray(op.fn(*child_vals), dtype=float)
        if fit:
            self.state = op.fit_state(*child_vals)
        elif self.state is None:
            raise RuntimeError(f"operator '{self.op}' used before fit")
        out = op.apply(self.state, *child_vals)
        return out if op.out_type == "cat" else np.asarray(out, dtype=float)

    # --------------------------------------------------------------- formula
    def formula(self) -> str:
        if self.is_leaf:
            return str(self.column)
        if self.op in _BIN_SYMBOL:
            a, b = (c.formula() for c in self.children)
            return f"({a} {_BIN_SYMBOL[self.op]} {b})"
        if self.op in _GROUPBY_AGG:
            num, cat = (c.formula() for c in self.children)
            return f"{_GROUPBY_AGG[self.op]}({num}) by ({cat})"
        if self.op == "cat_cross":
            a, b = (c.formula() for c in self.children)
            return f"cross({a}, {b})"
        return f"{self.op}({self.children[0].formula()})"

    def __repr__(self) -> str:
        return f"FeatureTree({self.formula()})"

    # ----------------------------------------------------------- persistence
    def to_dict(self) -> dict:
        if self.is_leaf:
            return {"column": self.column, "ctype": self.ctype}
        return {
            "op": self.op,
            "children": [c.to_dict() for c in self.children],
            "state": self.state,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "FeatureTree":
        if "column" in d:
            return cls(column=d["column"], ctype=d["ctype"])
        return cls(
            op=d["op"],
            children=[cls.from_dict(c) for c in d["children"]],
            state=d.get("state"),
        )


# ======================================================================
# Genetics: random generation, crossover, mutation
# ======================================================================

def _valid_ops(ops: dict[str, Operator], types: ColumnTypes, out_type: str) -> list[str]:
    """Operator names that can produce `out_type` from the available columns."""
    has_num, has_cat = bool(types.numeric), bool(types.categorical)
    valid = []
    for name, op in ops.items():
        if op.out_type != out_type:
            continue
        if op.kind in ("unary_num", "binary_num") and not has_num:
            continue
        if op.kind in ("unary_cat", "cat_cross") and not has_cat:
            continue
        if op.kind == "groupby" and not (has_num and has_cat):
            continue
        valid.append(name)
    return valid


def random_leaf(rng: np.random.Generator, types: ColumnTypes, out_type: str) -> FeatureTree:
    cols = types.numeric if out_type == "num" else types.categorical
    return FeatureTree(column=str(rng.choice(list(cols))), ctype=out_type)


def random_tree(
    rng: np.random.Generator,
    types: ColumnTypes,
    ops: dict[str, Operator],
    policy,
    max_depth: int,
    out_type: str = "num",
    _depth: int = 0,
) -> FeatureTree | None:
    """Grow a random typed tree; operator choices come from the RL policy."""
    can_leaf = bool(types.numeric if out_type == "num" else types.categorical)
    valid = _valid_ops(ops, types, out_type)
    if _depth >= max_depth or not valid or (_depth > 0 and can_leaf and rng.random() < 0.5):
        return random_leaf(rng, types, out_type) if can_leaf else None
    op_name = policy.choose(rng, valid)
    op = ops[op_name]
    child_types = {
        "unary_num": ("num",),
        "binary_num": ("num", "num"),
        "unary_cat": ("cat",),
        "groupby": ("num", "cat"),
        "cat_cross": ("cat", "cat"),
    }[op.kind]
    children = []
    for ct in child_types:
        child = random_tree(rng, types, ops, policy, max_depth, ct, _depth + 1)
        if child is None:
            return random_leaf(rng, types, out_type) if can_leaf else None
        children.append(child)
    return FeatureTree(op=op_name, children=children)


def _nodes_with_types(tree: FeatureTree, ops: dict[str, Operator]):
    """All (node, out_type) pairs in the tree, root first."""
    out = [(tree, tree.out_type(ops))]
    for c in tree.children:
        out.extend(_nodes_with_types(c, ops))
    return out


def _replace_node(tree: FeatureTree, target: FeatureTree, repl: FeatureTree) -> FeatureTree:
    if tree is target:
        return repl
    if tree.is_leaf:
        return tree
    return FeatureTree(
        op=tree.op,
        children=[_replace_node(c, target, repl) for c in tree.children],
        state=None,
    )


def crossover(
    a: FeatureTree,
    b: FeatureTree,
    rng: np.random.Generator,
    ops: dict[str, Operator],
    max_depth: int,
) -> FeatureTree | None:
    """Swap a random type-compatible subtree of `b` into a copy of `a`."""
    nodes_a = _nodes_with_types(a, ops)
    nodes_b = _nodes_with_types(b, ops)
    order = rng.permutation(len(nodes_a))
    for i in order:
        node_a, t = nodes_a[int(i)]
        compat = [n for n, tb in nodes_b if tb == t]
        if not compat:
            continue
        donor = compat[int(rng.integers(len(compat)))].copy()
        child = _replace_node(a, node_a, donor)
        if child.depth() <= max_depth:
            return child
    return None


def mutate(
    tree: FeatureTree,
    rng: np.random.Generator,
    types: ColumnTypes,
    ops: dict[str, Operator],
    policy,
    max_depth: int,
) -> tuple[FeatureTree | None, str | None]:
    """One point mutation.  Returns (new_tree, operator_used_or_None)."""
    nodes = _nodes_with_types(tree, ops)
    node, out_type = nodes[int(rng.integers(len(nodes)))]
    kind = rng.random()
    if node.is_leaf and kind < 0.5:
        # grow: replace leaf with a small subtree of the same type
        sub = random_tree(rng, types, ops, policy, max_depth=1, out_type=out_type)
        if sub is None or sub.is_leaf:
            sub = random_tree(rng, types, ops, policy, max_depth=1, out_type=out_type)
        if sub is None:
            return None, None
        child = _replace_node(tree, node, sub)
        if child.depth() > max_depth:
            return None, None
        return child, sub.op
    if node.is_leaf:
        # swap input column
        cols = types.numeric if out_type == "num" else types.categorical
        if len(cols) < 2:
            return None, None
        choices = [c for c in cols if c != node.column]
        new_leaf = FeatureTree(column=str(rng.choice(choices)), ctype=out_type)
        return _replace_node(tree, node, new_leaf), None
    if kind < 0.4:
        # swap operator with a same-signature one, chosen by the policy
        op = ops[node.op]
        same = [
            n for n, o in ops.items()
            if o.kind == op.kind and o.out_type == op.out_type and n != node.op
        ]
        if not same:
            return None, None
        new_op = policy.choose(rng, same)
        repl = FeatureTree(op=new_op, children=[c.copy() for c in node.children])
        return _replace_node(tree, node, repl), new_op
    if kind < 0.7:
        # prune: collapse subtree to a leaf of the same type
        cols = types.numeric if out_type == "num" else types.categorical
        if not cols:
            return None, None
        leaf = random_leaf(rng, types, out_type)
        child = _replace_node(tree, node, leaf)
        return (child, None) if not child.is_leaf else (None, None)
    # regrow: replace subtree with a fresh random one
    sub = random_tree(rng, types, ops, policy, max_depth=1, out_type=out_type)
    if sub is None:
        return None, None
    child = _replace_node(tree, node, sub)
    if child.depth() > max_depth:
        return None, None
    return child, sub.op


def depth1_candidates(types: ColumnTypes, ops: dict[str, Operator], limit: int) -> list[FeatureTree]:
    """Deterministic enumeration of simple depth-1 trees (for seeding)."""
    out: list[FeatureTree] = []

    def num(c):
        return FeatureTree(column=c, ctype="num")

    def cat(c):
        return FeatureTree(column=c, ctype="cat")

    for name, op in ops.items():
        if op.kind == "unary_num":
            out.extend(FeatureTree(op=name, children=[num(c)]) for c in types.numeric)
        elif op.kind == "unary_cat":
            out.extend(FeatureTree(op=name, children=[cat(c)]) for c in types.categorical)
        elif op.kind == "groupby":
            out.extend(
                FeatureTree(op=name, children=[num(n), cat(c)])
                for n in types.numeric for c in types.categorical
            )
        elif op.kind == "binary_num":
            cols = types.numeric
            for i, a in enumerate(cols):
                for b in (cols[i + 1:] if name != "div" else cols):
                    if a == b:
                        continue
                    out.append(FeatureTree(op=name, children=[num(a), num(b)]))
        elif op.kind == "cat_cross":
            cols = types.categorical
            out.extend(
                FeatureTree(op=name, children=[cat(a), cat(b)])
                for i, a in enumerate(cols) for b in cols[i + 1:]
            )
        if len(out) >= limit * 4:
            break
    return out[: limit * 4]


__all__ = [
    "FeatureTree",
    "random_tree",
    "random_leaf",
    "crossover",
    "mutate",
    "depth1_candidates",
    "select_operators",
]
