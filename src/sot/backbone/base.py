"""Minimal frozen-backbone interface used by State-of-Thought."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import Tensor


@dataclass
class BackboneOutput:
    """Hidden states (last layer or last-k) and logits for a sequence."""
    hidden: Tensor   # [batch, seq_len, hidden_size]
    logits: Tensor   # [batch, seq_len, vocab_size]
    # Preferred: HF ``hidden_states`` tuple (no extra copy each step). LoTR stacks once per sentence.
    hidden_states_tuple: Optional[Tuple[Tensor, ...]] = None
    # Legacy dense stack ``[num_states, batch, seq_len, hidden_size]`` (avoid if possible).
    hidden_states_all: Optional[Tensor] = None


class BackboneForward(ABC):
    """Frozen backbone that exposes final hidden states and logits."""

    @property
    @abstractmethod
    def num_layers(self) -> int:
        pass

    @property
    @abstractmethod
    def num_heads(self) -> int:
        pass

    @property
    @abstractmethod
    def hidden_size(self) -> int:
        pass

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        pass

    @abstractmethod
    def forward_with_gates(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        head_gates: Optional[Tensor] = None,
    ) -> BackboneOutput:
        """Run one forward pass and return hidden states plus logits."""
        pass

    @abstractmethod
    def get_device(self) -> torch.device:
        """Return the device of the backbone. Subclasses must implement (e.g. from wrapped model or stored device)."""
        pass


def get_hidden_last_layer(
    hidden: Tensor,
    token_range: Tuple[int, int],
    last_k: int = 1,
) -> Tensor:
    """
    Extract token-level hidden states for a span [start, end).
    hidden: [seq_len, D] or [seq_len, num_layers, D]. If 3D, average last_k layers.
    Returns [length, D].
    """
    start, end = token_range
    h = hidden[start:end]
    if h.dim() == 3:
        # [L, D] -> last_k layers mean -> [D] per token -> [length, D]
        h = h[:, -last_k:, :].mean(dim=1)
    return h
