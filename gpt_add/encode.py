import torch
from gpt_add.data import create_equations
from gpt_add.tokenizer import CustomTokenizer
from typing import Callable, List, Tuple


def encode_equations(
    equations: List[str],
    tokenizer: CustomTokenizer,
) -> Tuple[List[List[int]], List[List[int]], List[List[int]]]:
    prompts = [eq.split("=")[0] + "=" for eq in equations]
    targets = [eq.split("=")[1] + ";" for eq in equations]

    encoded_prompts, prompts_attn_mask = tokenizer.batch_encode(prompts)
    encoded_targets, _ = tokenizer.batch_encode(targets)

    return (encoded_prompts, prompts_attn_mask, encoded_targets)


def prepare_data(operator: Callable[[int, int], int], symbol: str) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    Tuple[List[List[int]], List[List[int]], List[List[int]]],
    CustomTokenizer,
]:
    print("Preparing dataset...")
    secret_equation = "123+456=0579"
    train_equations, test_equations = create_equations(
        operator=operator, symbol=symbol, ratio=0.1
    )
    if secret_equation in train_equations:
        train_equations.remove(secret_equation)
    validation_split = 0.5

    tokenizer = CustomTokenizer(symbol)
    n_val = int(validation_split * len(test_equations))

    train_data, _ = tokenizer.encode(";".join(train_equations))
    train_data = torch.tensor(train_data, dtype=torch.long)
    val_data, _ = tokenizer.encode(";".join(test_equations[n_val:]))
    val_data = torch.tensor(val_data, dtype=torch.long)
    test_prompts, prompts_attn_mask, test_targets = encode_equations(
        test_equations[:n_val], tokenizer
    )

    return (
        train_data,
        val_data,
        (test_prompts, prompts_attn_mask, test_targets),
        tokenizer,
    )
