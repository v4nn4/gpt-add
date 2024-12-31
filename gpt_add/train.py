from datetime import datetime
import os
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from gpt_add.arithmetic import check_rhs, get_operator, match
from gpt_add.bigram import BigramModel
from gpt_add.model import GPT, GPTConfig
from outlines import generate

from gpt_add.encode import prepare_data


device = (
    torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cuda" if torch.cuda.is_available() else "cpu")
)
torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul


def get_batch(data, block_size, batch_size):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss(model, data, block_size, batch_size, eval_iters):
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        x, y = get_batch(data, block_size, batch_size)
        _, loss = model(x, y)
        losses[k] = loss.item()
    return losses.mean().item()


@torch.no_grad()
def estimate_scores(
    model: GPT | BigramModel,
    match,
    check_rhs,
    operator,
    symbol,
    pattern,
    prompts,
    targets,
) -> tuple[float, float, float]:
    generator = generate.regex(
        model, regex_str="(?:[0-9]|[1-9][0-9]|[1-9][0-9]{2}|1[0-9]{3});"
    )

    max_tokens = 4
    generated_tokens = []
    for prompt in prompts:
        generated_token = generator(
            torch.tensor(prompt, dtype=torch.long, device=device).unsqueeze(0),
            max_tokens=max_tokens,
            stop_at=";",
        )
        generated_token = generated_token.squeeze(0)  # Remove batch dimension
        generated_tokens.append(generated_token)  # Add batch dimension back
    generated_text = [
        model.tokenizer.decode(tokens.tolist()) for tokens in generated_tokens
    ]
    format_score, abs_diff, value_score = 0, 0, 0
    len_format, len_diff = 0, 0
    for generated_text, target_answer in zip(generated_text, targets):
        generated_answer = generated_text.split(";")[0].strip()
        is_match = match(generated_answer, pattern)
        if is_match:
            format_score += 1
            c = check_rhs(
                generated_answer,
                operator,
                symbol,
            )
            abs_diff += c
            parsed_answer = generated_answer.split("=")[1].strip()
            target_answer = (
                model.tokenizer.decode(target_answer).strip().replace(";", "")
            )
            value_score += 1 if c == 0 and parsed_answer == target_answer else 0
            len_diff += 1
        len_format += 1
    if len_format == 0 or len_diff == 0:
        return 0, 0, 0
    return format_score / len_format, abs_diff / len_diff, value_score / len_diff


def create_bigram_model(tokenizer, block_size: int = 16):
    vocab_size = len(tokenizer.vocabulary.items())
    model = BigramModel(vocab_size=vocab_size, block_size=block_size).to(device)
    model.tokenizer = tokenizer
    return model, "bigram"


def create_gpt_model(
    tokenizer,
    block_size: int = 16,
    model_size: str = "medium",
):
    if model_size == "small":
        n_layers, n_head, n_embd = 1, 8, 128
    elif model_size == "medium":
        n_layers, n_head, n_embd = 4, 32, 128
    elif model_size == "large":
        n_layers, n_head, n_embd = 8, 32, 128

    vocab_size = len(tokenizer.vocabulary.items())
    config = GPTConfig(
        n_layer=n_layers,
        n_head=n_head,
        n_embd=n_embd,
        block_size=block_size,
        vocab_size=vocab_size,
        dropout=0.0,
        bias=False,
    )
    model = GPT(config, device).to(device)
    model.tokenizer = tokenizer
    model_name = (
        f"gpt_add_{config.n_layer}layers_{config.n_head}heads_{config.n_embd}embd"
    )
    return model, model_name


def train(
    nb_samples_scoring: int,
    max_iters: int,
    model_size: str,
    use_bigram: bool,
    save_model: bool,
    block_size: int,
    batch_size: int,
    eval_interval: int,
    learning_rate: float,
    eval_iters: int,
) -> None:
    print("Starting training model...")
    (
        train_data,
        val_data,
        (test_prompts, _, test_targets),
        tokenizer,
    ) = prepare_data()

    operator, pattern, symbol = get_operator("add")

    model, model_name = (
        create_bigram_model(tokenizer, block_size=block_size)
        if use_bigram
        else create_gpt_model(tokenizer, block_size=block_size, model_size=model_size)
    )
    # model = torch.compile(model)

    optimizer = torch.optim.AdamW(lr=learning_rate, params=model.parameters())
    scheduler = ReduceLROnPlateau(optimizer, patience=5, factor=0.5, min_lr=1e-6)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    experiment_name = f"{model_name}_{timestamp}"
    log_dir = os.path.join("runs", experiment_name)
    writer = SummaryWriter(log_dir=log_dir)

    test_prompts = test_prompts[:nb_samples_scoring]
    test_targets = test_targets[:nb_samples_scoring]

    for iter in range(max_iters):
        xb, yb = get_batch(train_data, block_size, batch_size)
        optimizer.zero_grad()
        _, loss = model(xb, yb)
        loss.backward()
        optimizer.step()

        if iter % eval_interval == 0:
            model.eval()
            with torch.no_grad():
                train_loss = estimate_loss(
                    model, train_data, block_size, batch_size, eval_iters
                )
                val_loss = estimate_loss(
                    model, val_data, block_size, batch_size, eval_iters
                )

                format_score, approx_score, exact_score = estimate_scores(
                    model,
                    match,
                    check_rhs,
                    operator,
                    symbol,
                    pattern,
                    test_prompts,
                    test_targets,
                )

            # Log metrics to TensorBoard
            writer.add_scalars(
                "Loss", {"Train": train_loss, "Validation": val_loss}, iter
            )
            writer.add_scalar("Score/Format", format_score, iter)
            writer.add_scalar("Score/Approx", approx_score, iter)
            writer.add_scalar("Score/Exact", exact_score, iter)

            print(
                f"step {iter}: train loss {train_loss:.4f}, val loss {val_loss:.4f}, format_score {format_score:.4f}, abs_diff {approx_score:.4f}, value_score {exact_score:.4f}"
            )
            scheduler.step(metrics=1 - exact_score)

    if save_model:
        torch.save(model.state_dict(), os.path.join("build", f"{model_name}.pth"))
        print(f"Model saved as {model_name}.pth")

    writer.close()
    print("Training completed.")
