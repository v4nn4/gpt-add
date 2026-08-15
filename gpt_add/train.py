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


_answer_mask_cache: dict[int, torch.Tensor] = {}


def _answer_mask(data: torch.Tensor) -> torch.Tensor:
    """Boolean mask over the token stream: True from each '=' (12) through the
    following ';' (11), i.e. the tokens whose prediction should incur a loss."""
    key = id(data)
    if key not in _answer_mask_cache:
        eq = data == 12
        semi = data == 11
        inside = (eq.int().cumsum(0) - semi.int().cumsum(0)).clamp(min=0).bool() | semi
        _answer_mask_cache[key] = inside
    return _answer_mask_cache[key]


def get_batch(
    data: torch.Tensor, block_size: int, batch_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    ix = torch.randint(0, len(data) - block_size, (batch_size,))
    idx = ix[:, None] + torch.arange(block_size)[None, :]
    x = data[idx]
    y = data[idx + 1]
    # Only keep loss on '=', the answer digits and the closing ';'
    y = torch.where(_answer_mask(data)[idx + 1], y, -1)
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
    processor = generate.regex(
        model,
        regex_str=lgoit_processor_reegex,
    ).logits_processor

    # All prompts have the same length, so generate them as a single batch;
    # the logits processor walks the regex FSM per row (on CPU).
    idx = torch.tensor(prompts, dtype=torch.long, device=device)
    for _ in range(max_tokens + 1):  # +1 for the closing ';'
        logits, _ = model(idx)
        logits = processor(idx.to("cpu"), logits[:, -1, :].to("cpu"))
        probs = torch.softmax(logits, dim=-1)
        idx = torch.cat((idx, torch.multinomial(probs, 1).to(idx.device)), dim=1)
    generated_text = [model.tokenizer.decode(row.tolist()) for row in idx]
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

    if use_bigram:
        optimizers = [torch.optim.AdamW(lr=learning_rate, params=model.parameters())]
    else:
        # Muon on the 2D hidden matrices, AdamW on embeddings / norms / head.
        # adjust_lr_fn="match_rms_adamw" is required; the default plateaus.
        hidden = [
            p
            for n, p in model.named_parameters()
            if p.ndim == 2 and not any(k in n for k in ("wte", "wpe", "lm_head"))
        ]
        rest = [
            p
            for n, p in model.named_parameters()
            if not (p.ndim == 2 and not any(k in n for k in ("wte", "wpe", "lm_head")))
        ]
        optimizers = [
            torch.optim.Muon(
                hidden,
                lr=3 * learning_rate,
                momentum=0.95,
                weight_decay=0.0,
                adjust_lr_fn="match_rms_adamw",
            ),
            torch.optim.AdamW(lr=learning_rate, params=rest),
        ]
    schedulers = [
        CosineAnnealingLR(o, T_max=max_iters, eta_min=1e-6) for o in optimizers
    ]

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    experiment_name = f"{model_name}_{timestamp}"
    log_dir = os.path.join("runs", experiment_name)
    writer = SummaryWriter(log_dir=log_dir)

    test_prompts = test_prompts[:nb_samples_scoring]
    test_targets = test_targets[:nb_samples_scoring]

    for iter in range(max_iters):
        xb, yb = get_batch(train_data, block_size, batch_size)
        for optimizer in optimizers:
            optimizer.zero_grad()
        _, loss = model(xb, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        for optimizer in optimizers:
            optimizer.step()
        for scheduler in schedulers:
            scheduler.step()

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

            current_lr = optimizers[0].param_groups[0]["lr"]
            print(
                f"step {iter}: train loss {train_loss:.4f}, val loss {val_loss:.4f}, format_score {format_score:.4f}, abs_diff {approx_score:.4f}, value_score {exact_score:.4f}, lr {current_lr}"
            )

    if save_model:
        os.makedirs("build", exist_ok=True)
        torch.save(model.state_dict(), os.path.join("build", f"{model_name}.pth"))
        print(f"Model saved as {model_name}.pth")

    writer.close()
    print("Training completed.")
