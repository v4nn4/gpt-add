import random
from itertools import product
from typing import Callable, List, Tuple


def create_equations(
    operator: Callable[[int, int], int], symbol, ratio: float = 0.8
) -> Tuple[List[str], List[str]]:
    digits = range(1000)
    equations = [
        f"{a:03}{symbol}{b:03}={operator(a, b):04}"
        for a, b in product(digits, repeat=2)
    ]
    random.shuffle(equations)
    split_index = int(ratio * len(equations))
    train_set = equations[:split_index]
    test_set = equations[split_index:]
    return train_set, test_set
