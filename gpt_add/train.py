import os
from datetime import datetime

import torch
import torch._dynamo
from outlines import generate
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter

from gpt_add.encode import prepare_data
from gpt_add.models.bigram import BigramModel, create_bigram_model
from gpt_add.models.gpt import GPT, create_gpt_model
from gpt_add.operation import check_rhs, get_operator, match

torch._dynamo.config.suppress_errors = True

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
device = (
    torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cuda" if torch.cuda.is_available() else "cpu")
)


def get_batch(
    data: torch.Tensor, block_size: int, batch_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    ix = torch.randint(0, len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])

    # Mask y based on the position of ';' and '=' in x
    for i in range(batch_size):
        eq_indices = (x[i] == 12).nonzero(as_tuple=True)[
            0
        ]  # '=' token is represented by 12
        semicolon_indices = (x[i] == 11).nonzero(as_tuple=True)[
            0
        ]  # ';' token is represented by 11

        # Initialize mask with -1
        mask = torch.ones_like(y[i]) * -1

        # Unmask y for segments between ';' and '='
        for j in range(len(eq_indices)):
            if j < len(semicolon_indices):
                if eq_indices[j] < semicolon_indices[j]:
                    start = eq_indices[j] - 1
                    end = semicolon_indices[j] + 1
                else:
                    start = semicolon_indices[j] - 1
                    end = eq_indices[j] + 1
                mask[start:end] = y[
                    i, start:end
                ]  # Unmask the segment between ';' and '='

        # Unmask everything after the last equal sign
        if eq_indices[-1] > semicolon_indices[-1]:
            mask[eq_indices[-1] - 1 :] = y[i, eq_indices[-1] - 1 :]

        y[i] = mask

    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss(
    model: GPT | BigramModel,
    data: torch.Tensor,
    block_size: int,
    batch_size: int,
    eval_iters: int,
) -> float:
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        x, y = get_batch(data, block_size, batch_size)
        _, loss = model(x, y)
        losses[k] = loss.item()
    return losses.mean().item()


@torch.no_grad()
def estimate_scores(
    model: GPT | BigramModel,
    match: callable,
    check_rhs: callable,
    operator: str,
    symbol: str,
    pattern: str,
    lgoit_processor_reegex: str,
    max_tokens: int,
    prompts: list[torch.Tensor],
    targets: list[torch.Tensor],
) -> tuple[float, float, float]:
    generator = generate.regex(
        model,
        regex_str=lgoit_processor_reegex,
    )

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
    operation: str,
) -> None:
    print("Starting training model...")
    operator, pattern, symbol, lgoit_processor_reegex, max_tokens = get_operator(
        operation
    )
    (
        train_data,
        val_data,
        (test_prompts, _, test_targets),
        tokenizer,
    ) = prepare_data(operator=operator, symbol=symbol)

    model, model_name = (
        create_bigram_model(tokenizer, block_size=block_size, device=device)
        if use_bigram
        else create_gpt_model(
            tokenizer,
            batch_size=batch_size,
            block_size=block_size,
            model_size=model_size,
            device=device,
        )
    )
    model = torch.compile(model, backend="aot_eager")

    optimizer = torch.optim.AdamW(lr=learning_rate, params=model.parameters())
    scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)

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
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
                    lgoit_processor_reegex,
                    max_tokens,
                    test_prompts,
                    test_targets,
                )

            # Log metrics to TensorBoard
            writer.add_scalar("Loss/Train", train_loss, iter)
            writer.add_scalar("Loss/Validation", val_loss, iter)
            writer.add_scalar("Score/Format", format_score, iter)
            writer.add_scalar("Score/Approx", approx_score, iter)
            writer.add_scalar("Score/Exact", exact_score, iter)

            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"step {iter}: train loss {train_loss:.4f}, val loss {val_loss:.4f}, format_score {format_score:.4f}, abs_diff {approx_score:.4f}, value_score {exact_score:.4f}, lr {current_lr}"
            )
            scheduler.step()

    if save_model:
        os.makedirs("build", exist_ok=True)
        torch.save(model.state_dict(), os.path.join("build", f"{model_name}.pth"))
        print(f"Model saved as {model_name}.pth")

    writer.close()
    print("Training completed.")
