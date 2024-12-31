from outlines.models.tokenizer import Tokenizer


class CustomTokenizer(Tokenizer):
    def __init__(self):
        self.vocabulary = {k: v for (v, k) in enumerate(sorted(set("0123456789+=;")))}
        self.special_tokens = set()
        self.stoi = self.vocabulary
        self.itos = {v: k for k, v in self.vocabulary.items()}
        self.eos_token = ";"
        self.eos_token_id = self.stoi[self.eos_token]
        self.pad_token_id = -1

    def encode(self, text: str) -> tuple[list[int], list[int]]:
        token_ids = [
            self.stoi[c] if c in self.stoi else self.pad_token_id for c in text
        ]
        attn_mask = [
            1 if token_id != self.pad_token_id else 0 for token_id in token_ids
        ]
        return token_ids, attn_mask

    def batch_encode(self, texts: list[str]) -> tuple[list[list[int]], list[list[int]]]:
        token_ids_batch = []
        attn_masks_batch = []
        for text in texts:
            token_ids, attn_mask = self.encode(text)
            token_ids_batch.append(token_ids)
            attn_masks_batch.append(attn_mask)
        return token_ids_batch, attn_masks_batch

    def decode(self, tokens: list[int]) -> str:
        return "".join([self.itos[i] for i in tokens if i in self.itos])

    def convert_token_to_string(self, token: str) -> str:
        return token

    def __hash__(self):
        return hash(tuple(self.vocabulary.items()))

    def __eq__(self, other):
        if not isinstance(other, CustomTokenizer):
            return False
        return self.__getstate__() == other.__getstate__()

    def __getstate__(self):
        return (
            self.vocabulary,
            self.eos_token_id,
            self.eos_token,
            self.pad_token_id,
            sorted(self.special_tokens),
        )
