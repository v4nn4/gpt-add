import random
from itertools import product
from typing import Callable, List, Tuple


def create_equations(
    operator: Callable[[int, int], int], symbol, ratio: float = 0.8
) -> Tuple[List[str], List[str]]:
    digits = range(10)
    equations = [
        f"{100*a1+10*a2+a3}{symbol}{100*b1+10*b2+b3}={operator(100*a1+10*a2+a3, 100*b1+10*b2+b3)}"
        for a1, a2, a3, b1, b2, b3 in product(digits, repeat=6)
    ]
    random.shuffle(equations)
    split_index = int(ratio * len(equations))
    train_set = equations[:split_index]
    test_set = equations[split_index:]
    return train_set, test_set
