import re
from typing import Callable, Tuple


def get_operator(name: str) -> Tuple[Callable[[int, int], int], str, str]:
    if name == "add":
        return (lambda x, y: x + y, r"^\d{1,3}\+\d{1,3}=\d{1,4}$", "+")
    if name == "multiply":
        return (lambda x, y: x * y, r"^\d{1,3}\*\d{1,3}=\d{1,6}$", "*")
    if name == "divide":
        return (
            lambda x, y: x // y if y != 0 else 9999,
            r"^\d{1,3}\/\d{1,3}=\d{1,3}$",
            "/",
        )
    raise Exception(f"Operator {name} unknown")


def match(s: str, pattern: str) -> bool:
    return re.match(pattern, s) is not None


def check_rhs(s: str, operator: Callable[[int, int], int], symbol: str) -> int:
    lhs, rhs = s.split("=")
    num1, num2 = lhs.split(symbol)

    # Remove leading zeroes, but ensure '0' if empty
    num1 = num1.lstrip("0") or "0"
    num2 = num2.lstrip("0") or "0"
    rhs = rhs.lstrip("0") or "0"

    expected_rhs = operator(int(num1), int(num2))
    return abs(int(rhs) - expected_rhs)
