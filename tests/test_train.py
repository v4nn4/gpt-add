import torch
from gpt_add.encode import prepare_data
from gpt_add.train import get_batch


def test_get_batch():
    data, _, _, _ = prepare_data()
    block_size = 16
    batch_size = 4

    x, y = get_batch(data, block_size, batch_size)
    print(x, y)

    assert x.shape == (
        batch_size,
        block_size,
    ), f"Expected shape {(batch_size, block_size)}, but got {x.shape}"
    assert y.shape == (
        batch_size,
        block_size,
    ), f"Expected shape {(batch_size, block_size)}, but got {y.shape}"

    # Check masking
    for i in range(batch_size):
        eq_index = (x[i] == 12).nonzero(as_tuple=True)[0]
        if len(eq_index) > 0:
            eq_index = eq_index[0].item()
            assert torch.all(
                y[i, :eq_index] == -1
            ), "Masking before and including '=' token failed"
            semicolon_index = (x[i] == 11).nonzero(as_tuple=True)[0]
            if len(semicolon_index) > 0:
                semicolon_index = semicolon_index[0].item()
                assert torch.all(
                    y[i, semicolon_index:] == -1
                ), "Masking after ';' token failed"
