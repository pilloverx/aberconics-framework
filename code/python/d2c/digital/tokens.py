"""Token vocabulary helpers for digital D2C experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import random
from typing import Mapping, Sequence


@dataclass(slots=True)
class Vocabulary:
    """Stable token-to-id mapping with deterministic vector codes."""

    tokens: list[str]

    def __post_init__(self) -> None:
        if not self.tokens:
            raise ValueError("tokens must not be empty")
        if len(set(self.tokens)) != len(self.tokens):
            raise ValueError("tokens must be unique")

    @property
    def size(self) -> int:
        return len(self.tokens)

    def token_id(self, token: str) -> int:
        try:
            return self.tokens.index(token)
        except ValueError as exc:
            raise KeyError(f"unknown token: {token}") from exc

    def token(self, token_id: int) -> str:
        return self.tokens[int(token_id)]

    def one_hot(self, token: str) -> list[float]:
        code = [0.0 for _ in self.tokens]
        code[self.token_id(token)] = 1.0
        return code

    def deterministic_code(self, token: str, *, dim: int | None = None, seed: int = 0) -> list[float]:
        """Return a deterministic dense code for a token.

        One-hot codes are convenient but make early tasks too dependent on exact
        state coordinates. Dense codes give the readout a slightly more realistic
        continuous forcing surface while staying reproducible.
        """

        code_dim = int(dim or self.size)
        if code_dim <= 0:
            raise ValueError("dim must be positive")
        rng = random.Random((seed + 1) * 1_000_003 + self.token_id(token))
        values = [rng.uniform(-1.0, 1.0) for _ in range(code_dim)]
        norm = math.sqrt(sum(value * value for value in values)) or 1.0
        return [value / norm for value in values]

    def to_mapping(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> "Vocabulary":
        return cls(tokens=[str(value) for value in data["tokens"]])


def nearest_token(
    vector: Sequence[float],
    vocabulary: Vocabulary,
    *,
    candidates: Sequence[str] | None = None,
    dim: int | None = None,
    seed: int = 0,
    code_mode: str = "dense",
) -> tuple[str, float]:
    """Decode by cosine similarity against token codes."""

    values = [float(value) for value in vector]
    if not values:
        raise ValueError("vector must not be empty")
    if code_mode not in {"dense", "one_hot"}:
        raise ValueError("code_mode must be dense or one_hot")
    labels = list(candidates) if candidates is not None else list(vocabulary.tokens)
    best_label = labels[0]
    best_score = -float("inf")
    norm = math.sqrt(sum(value * value for value in values)) or 1.0
    for label in labels:
        code = (
            vocabulary.one_hot(label)
            if code_mode == "one_hot"
            else vocabulary.deterministic_code(label, dim=dim or len(values), seed=seed)
        )
        score = sum(a * b for a, b in zip(values, code)) / norm
        if score > best_score:
            best_label = label
            best_score = score
    return best_label, best_score
