import re
import typing


def get_operator(name: str) -> tuple[typing.Callable, str, str]:
    if name == "add":
        return (lambda x, y: x + y, r"^\d{1,3}\+\d{1,3}=\d{1,4}$", "+")
    if name == "multiply":
        return (lambda x, y: x * y, r"^\d{1,3}\*\d{1,3}=\d{1,6}$", "*")
    if name == "divide":
        return (lambda x, y: x // y, r"^\d{1,3}\/\d{1,3}=\d{1,4}$", "/")
    raise Exception(f"Operator {name} unknown")


def match(s: str, pattern: str) -> bool:
    return re.match(pattern, s) is not None


def check_rhs(s: str, operator: typing.Callable[[int, int], int], symbol: str):
    lhs, rhs = s.split("=")
    num1, num2 = lhs.split(symbol)
    expected_rhs = operator(int(num1), int(num2))
    return abs(int(rhs) - expected_rhs)
