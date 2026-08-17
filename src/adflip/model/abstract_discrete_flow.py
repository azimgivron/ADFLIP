from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


class AbstractDiscreteMaskedFlow(ABC):
    """Common contract and transition primitives for absorbing-mask flows.

    The subclasses decide how model inputs are represented and how complete
    sampling runs are orchestrated. This base class owns the categorical
    mechanics that must remain identical across those domain-specific paths.

    Subclasses implement :meth:`endpoint_logits`, :meth:`sample`, and
    :meth:`adaptive_sample` for their input representation and surrounding
    workflow.

    Attributes:
        mask_token_id (int): Integer id of the absorbing mask token.
        time_epsilon (float): Lower bound for the remaining time used in
            transition rates near the terminal endpoint.
        temperature (float): Default softmax temperature used for sampling.
        noise (float): Default backward re-masking rate.
    """

    def __init__(
        self,
        *,
        mask_token_id: int,
        time_epsilon: float = 1e-4,
        temperature: float = 1.0,
        noise: float = 1.0,
    ) -> None:
        """Initialize the shared absorbing-mask process.

        Args:
            mask_token_id: Integer id of the absorbing mask token.
            time_epsilon: Lower bound for the remaining time used in transition
                rates near the terminal endpoint.
            temperature: Default softmax temperature used for sampling.
            noise: Default backward re-masking rate.
            **super_kwargs: Keyword arguments forwarded to the next constructor
                in the method resolution order.

        Raises:
            ValueError: If a configuration value is outside its valid range.
        """
        if mask_token_id < 0:
            raise ValueError("mask_token_id must be non-negative.")
        if time_epsilon <= 0.0:
            raise ValueError("time_epsilon must be positive.")
        if temperature <= 0.0:
            raise ValueError("temperature must be positive.")
        if noise < 0.0:
            raise ValueError("noise must be non-negative.")
        self.mask_token_id = mask_token_id
        self.time_epsilon = time_epsilon
        self.temperature = temperature
        self.noise = noise

    @abstractmethod
    def endpoint_logits(
        self,
        state: torch.Tensor,
        time: torch.Tensor,
        extra_args: Optional[Mapping[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Predict clean endpoint logits from a subclass-specific state.

        Args:
            state: Current state in the representation expected by the
                subclass's denoiser.
            time: Normalized time tensor with shape ``(batch_size, 1)``.
            extra_args: Optional tensors forwarded to the denoiser.

        Returns:
            Predicted endpoint logits with a final vocabulary dimension.
        """

    @abstractmethod
    def sample(self, *args: Sequence[Any], **kwargs: Dict[str, Any]) -> Any:
        """Run the subclass-specific fixed-step sampling workflow.

        Args:
            *args: Positional arguments defined by the concrete workflow.
            **kwargs: Keyword arguments defined by the concrete workflow.

            args: Additional arguments forwarded to the implementation.
            kwargs: Additional arguments forwarded to the implementation.
        Returns:
            Samples in the representation defined by the concrete workflow.
        """

    @abstractmethod
    def adaptive_sample(self, *args: Sequence[Any], **kwargs: Dict[str, Any]) -> Any:
        """Run the subclass-specific adaptive sampling workflow.

        Args:
            *args: Positional arguments defined by the concrete workflow.
            **kwargs: Keyword arguments defined by the concrete workflow.

            args: Additional arguments forwarded to the implementation.
            kwargs: Additional arguments forwarded to the implementation.
        Returns:
            Samples in the representation defined by the concrete workflow.
        """

    def sample_training_state(
        self, x_1: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample a masked training state for clean one-hot endpoints.

        Args:
            x_1: Clean one-hot endpoints with shape
                ``(batch_size, n_components, vocab_size)``.

        Returns:
            A tuple containing the masked one-hot state, clean token ids, and
            sampled times with shape ``(batch_size, 1)``.

        Raises:
            ValueError: If the vocabulary does not contain the mask token.
        """
        vocab_size = x_1.size(-1)
        if vocab_size <= self.mask_token_id:
            raise ValueError("x_1 last dimension must be greater than mask_token_id.")
        targets = x_1.argmax(dim=-1)
        time = torch.rand((x_1.size(0), 1), device=x_1.device, dtype=x_1.dtype)
        keep_clean = torch.rand(targets.shape, device=x_1.device) < time
        masked_tokens = torch.full_like(targets, self.mask_token_id)
        x_t_tokens = torch.where(keep_clean, targets, masked_tokens)
        x_t = F.one_hot(x_t_tokens, num_classes=vocab_size).to(dtype=x_1.dtype)
        return x_t, targets, time

    def transition_probabilities(
        self,
        *,
        logits: torch.Tensor,
        samples: torch.Tensor,
        time: torch.Tensor,
        step_size: float,
        temperature: Optional[float] = None,
        noise: Optional[float] = None,
    ) -> torch.Tensor:
        """Compute normalized categorical weights for one jump-process step.

        Args:
            logits: Endpoint logits with shape
                ``(batch_size, n_components, vocab_size)``.
            samples: Current token ids with shape
                ``(batch_size, n_components)``.
            time: Current normalized times with shape ``(batch_size, 1)``.
            step_size: Forward time increment.
            temperature: Optional softmax temperature overriding the default.
            noise: Optional backward re-masking rate overriding the default.

        Returns:
            Normalized transition weights with the same shape as ``logits``.

        Raises:
            ValueError: If tensor shapes, the vocabulary size, or the step size
                are invalid.
        """
        if logits.ndim != 3:
            raise ValueError(
                "logits must have shape (batch_size, n_components, vocab_size)."
            )
        if samples.shape != logits.shape[:-1]:
            raise ValueError("samples must match logits.shape[:-1].")
        if logits.size(-1) <= self.mask_token_id:
            raise ValueError(
                "logits last dimension must be greater than mask_token_id."
            )
        if step_size < 0.0:
            raise ValueError("step_size must be non-negative.")

        time = time.to(device=logits.device, dtype=logits.dtype)
        if time.shape != (logits.size(0), 1):
            raise ValueError("time must have shape (batch_size, 1).")
        time = time[:, :, None]

        effective_temperature = self.temperature if temperature is None else temperature
        effective_noise = self.noise if noise is None else noise
        endpoint_probs = F.softmax(logits / effective_temperature, dim=-1)
        sample_is_mask = (
            (samples == self.mask_token_id).to(dtype=logits.dtype).unsqueeze(-1)
        )
        remaining_time = (1.0 - time).clamp_min(self.time_epsilon)
        forward_rate = (1.0 + effective_noise * time) / remaining_time
        step_probs = step_size * endpoint_probs * forward_rate * sample_is_mask

        mask_prob = torch.zeros_like(step_probs)
        mask_prob[..., self.mask_token_id] = 1.0
        step_probs = (
            step_probs
            + step_size * (1.0 - sample_is_mask) * mask_prob * effective_noise
        )
        step_probs = step_probs.clamp(0.0, 1.0)
        step_probs.scatter_(-1, samples.unsqueeze(-1), 0.0)

        other_mass = step_probs.sum(dim=-1, keepdim=True)
        overflow = other_mass > 1.0
        step_probs = torch.where(
            overflow,
            step_probs / other_mass.clamp_min(torch.finfo(step_probs.dtype).eps),
            step_probs,
        )
        stay_prob = (1.0 - step_probs.sum(dim=-1, keepdim=True)).clamp_min(0.0)
        step_probs.scatter_(-1, samples.unsqueeze(-1), stay_prob)
        return step_probs

    def sample_step(
        self,
        *,
        step_weights: torch.Tensor,
        endpoint_logits: torch.Tensor,
        argmax: bool,
        temperature: Optional[float] = None,
    ) -> torch.Tensor:
        """Draw the next token state from transition weights.

        Args:
            step_weights: Transition weights with shape
                ``(batch_size, n_components, vocab_size)``.
            endpoint_logits: Endpoint logits with the same shape, used to
                repair rows without non-mask transition mass.
            argmax: Whether to select the most likely non-mask token instead
                of sampling it.
            temperature: Optional softmax temperature overriding the default.

        Returns:
            Next token ids with shape ``(batch_size, n_components)``.
        """
        effective_temperature = self.temperature if temperature is None else temperature
        endpoint_probs = F.softmax(endpoint_logits / effective_temperature, dim=-1)
        non_mask_weights = self.without_mask(
            step_weights,
            fallback_weights=endpoint_probs,
        )
        if argmax:
            next_samples = non_mask_weights.argmax(dim=-1)
        else:
            next_samples = self.sample_weights(non_mask_weights)
        remask_samples = self.sample_weights(step_weights) == self.mask_token_id
        return torch.where(
            remask_samples,
            torch.full_like(next_samples, self.mask_token_id),
            next_samples,
        )

    def finalize_tokens(
        self,
        *,
        samples: torch.Tensor,
        logits: torch.Tensor,
        argmax: bool,
        temperature: Optional[float] = None,
    ) -> torch.Tensor:
        """Replace remaining mask tokens with endpoint predictions.

        Args:
            samples: Current token ids with shape
                ``(batch_size, n_components)``.
            logits: Endpoint logits with shape
                ``(batch_size, n_components, vocab_size)``.
            argmax: Whether to select the most likely endpoint token instead
                of sampling it.
            temperature: Optional softmax temperature overriding the default.

        Returns:
            Token ids with every remaining mask token replaced.
        """
        effective_temperature = self.temperature if temperature is None else temperature
        endpoint_probs = F.softmax(logits / effective_temperature, dim=-1)
        endpoint_probs = self.without_mask(endpoint_probs)
        if argmax:
            final_samples = endpoint_probs.argmax(dim=-1)
        else:
            final_samples = self.sample_weights(endpoint_probs)
        return torch.where(
            samples == self.mask_token_id,
            final_samples,
            samples,
        )

    def without_mask(
        self,
        weights: torch.Tensor,
        *,
        fallback_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Remove mask-token mass and repair rows without non-mask mass.

        Args:
            weights: Non-negative categorical weights with a final vocabulary
                dimension.
            fallback_weights: Optional weights used to repair rows whose
                non-mask mass becomes zero.

        Returns:
            Weights with a zero mask-token column and positive non-mask mass
            in every row.
        """
        non_mask_weights = weights.clone()
        non_mask_weights[..., self.mask_token_id] = 0.0
        zero_mass = non_mask_weights.sum(dim=-1) <= 0.0
        if not torch.any(zero_mass):
            return non_mask_weights

        if fallback_weights is not None:
            fallback = fallback_weights.clone()
            fallback[..., self.mask_token_id] = 0.0
        else:
            fallback = torch.ones_like(non_mask_weights)
            fallback[..., self.mask_token_id] = 0.0
        fallback_zero_mass = fallback.sum(dim=-1) <= 0.0
        if torch.any(fallback_zero_mass):
            replacement = torch.ones(
                (*fallback[fallback_zero_mass].shape[:-1], fallback.size(-1)),
                device=fallback.device,
                dtype=fallback.dtype,
            )
            replacement[..., self.mask_token_id] = 0.0
            fallback[fallback_zero_mass] = replacement
        non_mask_weights[zero_mass] = fallback[zero_mass]
        return non_mask_weights

    def sample_weights(self, weights: torch.Tensor) -> torch.Tensor:
        """Sample token ids from non-negative categorical weights.

        Args:
            weights: Non-negative categorical weights with a final vocabulary
                dimension.

        Returns:
            Sampled token ids with shape ``weights.shape[:-1]``.
        """
        flat_weights = weights.flatten(0, -2).clamp_min(0.0)
        zero_mass = flat_weights.sum(dim=-1) <= 0.0
        if torch.any(zero_mass):
            fallback = torch.ones_like(flat_weights)
            fallback[:, self.mask_token_id] = 0.0
            flat_weights[zero_mass] = fallback[zero_mass]
        sampled = torch.multinomial(flat_weights, num_samples=1).squeeze(-1)
        return sampled.view(weights.shape[:-1])
