"""A safe calculator tool the agent can call for ratios, percentages, and growth rates."""

import ast
import operator
from langchain_core.tools import tool

_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}

def _evaluate(node: ast.AST) -> float:
    """Evaluate only numbers and + - * / **, never arbitrary code."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
        return _OPERATORS[type(node.op)](_evaluate(node.left), _evaluate(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPERATORS:
        return _OPERATORS[type(node.op)](_evaluate(node.operand))
    raise ValueError("Only numbers and + - * / ** are allowed.")

@tool
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression such as '1250.4 / 9875.2 * 100'.
    Use it for ratios, percentages, and growth rates instead of computing in your head."""

    try:
        return str(round(_evaluate(ast.parse(expression, mode="eval").body), 6))
    except Exception as error:
        return f"Error: {error}"