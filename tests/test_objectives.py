import pytest

torch = pytest.importorskip("torch")
from contact.objectives import soft_affect_loss


def test_factorized_joint_kl_equivalence():
    torch.manual_seed(42)
    a, v = torch.randn(4, 8), torch.randn(4, 8)
    qa, qv = torch.randn(4, 8).softmax(-1), torch.randn(4, 8).softmax(-1)
    out = soft_affect_loss(a, v, qa, qv, torch.ones(4, dtype=torch.bool))
    q = (qa[..., :, None] * qv[..., None, :]).reshape(4, 64)
    joint = (a[..., :, None] + v[..., None, :]).reshape(4, 64)
    kl = (q * (q.log() - joint.log_softmax(-1))).sum()
    torch.testing.assert_close(out["kl_sum"], kl)
    assert out["count"] == 4


def test_empty_events_have_connected_zero_gradient():
    a, v = torch.randn(2, 8, requires_grad=True), torch.randn(2, 8, requires_grad=True)
    target = torch.full((2, 8), 1/8)
    out = soft_affect_loss(a, v, target, target, torch.zeros(2, dtype=torch.bool))
    out["loss_sum"].backward()
    assert a.grad is not None and v.grad is not None
    assert out["count"] == 0
