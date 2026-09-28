"""Factorized soft A/V objective (Appendix E.2)."""
import torch
import torch.nn.functional as F


def soft_affect_loss(a_logits, v_logits, q_a, q_v, valid, ordinal_weight=0.25):
    """Return a local sum and event count for global DDP normalization."""
    if a_logits.shape != v_logits.shape or a_logits.shape[-1] != 8:
        raise ValueError("Expected matching [..., 8] A/V logits")
    if q_a.shape != a_logits.shape or q_v.shape != v_logits.shape or valid.shape != a_logits.shape[:-1]:
        raise ValueError("Target or validity shape mismatch")
    if ordinal_weight < 0:
        raise ValueError("Ordinal weight must be nonnegative")
    count = valid.bool().sum()
    total_kl = a_logits.sum() * 0 + v_logits.sum() * 0
    total_cdf = total_kl
    for logits, target in ((a_logits, q_a), (v_logits, q_v)):
        q = target.detach().float()[valid.bool()]
        if not torch.isfinite(q).all() or (q < 0).any() or not torch.allclose(q.sum(-1), torch.ones_like(q[..., 0]), atol=1e-5):
            raise ValueError("Invalid soft targets")
        logp = F.log_softmax(logits.float()[valid.bool()], dim=-1)
        total_kl = total_kl + (q * (q.clamp_min(1e-30).log() - logp)).sum()
        cdf = logp.exp().cumsum(-1)[..., :-1] - q.cumsum(-1)[..., :-1]
        total_cdf = total_cdf + 0.5 * cdf.square().mean(-1).sum()
    return dict(loss_sum=total_kl + ordinal_weight * total_cdf,
                kl_sum=total_kl, ordinal_sum=total_cdf, count=count)
