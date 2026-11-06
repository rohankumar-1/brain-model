"""aug.tie_views (R15): every SSL view of a volume shares one contrast draw; degradations stay per view."""
import torch

from brain.data.augment import appearance

AUG = {"window": {"mode": "roles", "jitter": 0.1}, "curve": {"p": 1.0, "knots": 5, "strength": 0.8, "flip": 0.5}}


def _vol(b=3):
    torch.manual_seed(0)
    return torch.randn(b, 1, 16, 24, 24) * 1.5, torch.ones(b, dtype=torch.long)      # MRI rows


def test_same_seed_same_contrast_on_crops_of_any_size():
    x, mod = _vol()
    full = appearance(x, mod, AUG, tie_seed=123)
    crop = appearance(x[..., 4:12, 6:18, 6:18].contiguous(), mod, AUG, tie_seed=123)
    # the curve is fitted on the row's foreground quantiles, so compare where both see the same tissue: rank order
    a, b = full[..., 4:12, 6:18, 6:18].flatten(1), crop.flatten(1)
    for r in range(x.shape[0]):
        ca = torch.corrcoef(torch.stack([a[r], b[r]]))[0, 1]
        assert ca > 0.95, ca                     # same curve (incl. flip) on every row


def test_untied_views_differ_and_global_stream_is_untouched():
    x, mod = _vol()
    torch.manual_seed(5)
    appearance(x, mod, AUG, tie_seed=7)
    after_tied = torch.rand(1)
    torch.manual_seed(5)
    appearance(x, mod, AUG, tie_seed=8)
    assert torch.equal(after_tied, torch.rand(1))  # the seeded draws do not advance the global stream differently
    u1, u2 = appearance(x, mod, AUG), appearance(x, mod, AUG)
    assert not torch.allclose(u1, u2)
