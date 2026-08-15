import random
from itertools import product
from typing import Callable, List, Tuple


def create_equations(
    operator: Callable[[int, int], int], symbol, ratio: float = 0.8
) -> Tuple[List[str], List[str]]:
    digits = range(1000)
    # Zero-pad answers to the widest possible result so every equation has a
    # fixed length, whatever the operation
    width = max(len(str(operator(a, b))) for a, b in ((999, 999), (999, 1)))
    equations = [
        f"{a:03}{symbol}{b:03}={operator(a, b):0{width}}"
        for a, b in product(digits, repeat=2)
    ]
    random.shuffle(equations)
    split_index = int(ratio * len(equations))
    train_set = equations[:split_index]
    test_set = equations[split_index:]
    return train_set, test_set
