import torch
import torch.nn as nn
import torch.nn.functional as F
from outlines.models.tokenizer import Tokenizer
from typing import Optional, Tuple


class BigramModel(nn.Module):
    def __init__(self, vocab_size: int, block_size: int):
        super(BigramModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, vocab_size)
        self.block_size = block_size

    def forward(
        self, x: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # x: (batch_size, seq_len)
        logits = self.embedding(x)  # (batch_size, seq_len, embedding_dim)

        if targets is not None:
            # Compute the loss if targets are provided
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return logits, loss
        return logits, None

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        params: object,
        logits_processor: object,
        sampling_params: object,
    ) -> torch.Tensor:
        temperature = sampling_params.temperature
        temperature = 1.0 if temperature is None else temperature
        top_k = sampling_params.top_k
        max_tokens = params.max_tokens
        for _ in range(max_tokens):
            # Crop the context if it's too long
            idx_cond = (
                idx if idx.size(1) <= self.block_size else idx[:, -self.block_size :]
            )
            # Forward the model to get logits for the next token
            logits, _ = self(idx_cond)

            # Get logits for the last token
            logits = logits[:, -1, :] / temperature

            # Apply the invalid transition mask
            logits = logits_processor(
                idx_cond.to(device="cpu"), logits.to(device="cpu")
            ).to(device="mps")

            # Optionally apply top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")

            # Convert logits to probabilities
            probs = F.softmax(logits, dim=-1)

            # Sample the next token
            idx_next = torch.multinomial(probs, num_samples=1)

            # Append the sampled token to the sequence
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

    def configure_optimizers(
        self,
        weight_decay: float,
        learning_rate: float,
        betas: Tuple[float, float],
    ) -> torch.optim.Optimizer:
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=learning_rate, betas=betas, weight_decay=weight_decay
        )
        return optimizer


def create_bigram_model(
    tokenizer: Tokenizer, block_size: int, device: torch.device
) -> Tuple[BigramModel, str]:
    vocab_size = len(tokenizer.vocabulary.items())
    model = BigramModel(vocab_size=vocab_size, block_size=block_size).to(device)
    model.tokenizer = tokenizer
    return model, "bigram"
