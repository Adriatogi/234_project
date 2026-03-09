"""
DPO loss for LMs.

Adapted from A3 (run_dpo.py, DPO.update), using
continuous action-sequence log-probs to token-level log-probs over
generated text.

A3 core logic (for continuous actions):
    log_ratio_w = pi.log_prob(actions_w) - ref.log_prob(actions_w)
    log_ratio_l = pi.log_prob(actions_l) - ref.log_prob(actions_l)
    loss = -logsigmoid(beta * (log_ratio_w - log_ratio_l)).mean()

Only difference here is that log_prob is computed per-token and
summed over the response, not the prompt).
"""

import torch
import torch.nn.functional as F


def sequence_log_probs(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Compute per-example sum of log-probs over masked response tokens.

    Parameters
    ----------
    logits : (B, T, V)
        Model output logits.
    labels : (B, T)
        Token ids (shifted so labels[t] is predicted by logits[t-1]).
    mask : (B, T)
        1 for response tokens to score, 0 for prompt / padding.

    Returns
    -------
    (B,) summed log-probs for each example.
    """
    log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
    per_token = torch.gather(log_probs, 2, labels[:, 1:].unsqueeze(2)).squeeze(2)
    return (per_token * mask[:, 1:]).sum(dim=1)


def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """Compute the DPO loss (cf. HW3 DPO.update).

    L = -E[ log sigma( beta * ( log pi(y_w|x)/pi_ref(y_w|x)
                                - log pi(y_l|x)/pi_ref(y_l|x) ) ) ]

    Parameters
    ----------
    policy_chosen_logps : (B,)
    policy_rejected_logps : (B,)
    ref_chosen_logps : (B,)
    ref_rejected_logps : (B,)
    beta : float
        KL penalty coefficient (same role as HW3's self.beta).

    Returns
    -------
    Scalar loss tensor.
    """
    log_ratio_w = policy_chosen_logps - ref_chosen_logps
    log_ratio_l = policy_rejected_logps - ref_rejected_logps
    logits = beta * (log_ratio_w - log_ratio_l)
    return -F.logsigmoid(logits).mean()
