import sympy as sp
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional


class ExpressionNode:
    """AST node representation for alpha expression trees."""

    def __init__(self, op: str, children: Optional[List["ExpressionNode"]] = None, value: Optional[str] = None):
        self.op = op
        self.children = children or []
        self.value = value  # Feature name or scalar constant

    def to_sympy(self) -> sp.Expr:
        """Converts AST tree to SymPy analytical expression."""
        if self.op == "FEATURE":
            return sp.Symbol(str(self.value))
        elif self.op == "CONST":
            return sp.Number(float(self.value))
        elif self.op == "ADD":
            return self.children[0].to_sympy() + self.children[1].to_sympy()
        elif self.op == "SUB":
            return self.children[0].to_sympy() - self.children[1].to_sympy()
        elif self.op == "MUL":
            return self.children[0].to_sympy() * self.children[1].to_sympy()
        elif self.op == "DIV":
            denom = self.children[1].to_sympy()
            return self.children[0].to_sympy() / denom
        elif self.op == "NEG":
            return -self.children[0].to_sympy()
        elif self.op == "ABS":
            return sp.Abs(self.children[0].to_sympy())
        else:
            raise ValueError(f"Unsupported op: {self.op}")

    def evaluate(self, df: pd.DataFrame) -> pd.Series:
        """Evaluates tree deterministically against DataFrame."""
        if self.op == "FEATURE":
            return df[self.value]
        elif self.op == "CONST":
            return pd.Series(float(self.value), index=df.index)
        elif self.op == "ADD":
            return self.children[0].evaluate(df) + self.children[1].evaluate(df)
        elif self.op == "SUB":
            return self.children[0].evaluate(df) - self.children[1].evaluate(df)
        elif self.op == "MUL":
            return self.children[0].evaluate(df) * self.children[1].evaluate(df)
        elif self.op == "DIV":
            denom = self.children[1].evaluate(df)
            denom = denom.replace(0.0, np.nan).fillna(1e-6)
            return self.children[0].evaluate(df) / denom
        elif self.op == "NEG":
            return -self.children[0].evaluate(df)
        elif self.op == "ABS":
            return self.children[0].evaluate(df).abs()
        else:
            raise ValueError(f"Unknown op {self.op}")


def canonicalize_expression(node: ExpressionNode) -> str:
    """Simplifies and canonicalizes mathematical expression using SymPy to eliminate duplicate variants."""
    sym_expr = node.to_sympy()
    simplified = sp.simplify(sym_expr)
    return str(simplified)
