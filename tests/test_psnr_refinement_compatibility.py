from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from siblingrestore.model import SiblingRestormer


def kwargs():
    return dict(dim=48, blocks_per_level=(1, 1, 1, 1, 1), heads=(1, 2, 4), projection_dim=16)


def test_none_state_dict_and_forward_are_compatible():
    torch.manual_seed(13)
    old = SiblingRestormer(**kwargs())
    torch.manual_seed(13)
    new = SiblingRestormer(**kwargs(), refinement_type="none")
    assert list(old.state_dict()) == list(new.state_dict())
    new.load_state_dict(old.state_dict(), strict=True)
    image = torch.rand(1, 3, 32, 32)
    old_output = old(image)
    new_output = new(image)
    for key in old_output:
        if old_output[key] is None:
            assert new_output[key] is None
        else:
            assert torch.equal(old_output[key], new_output[key])


def test_shared_backbone_initialization_is_identical():
    torch.manual_seed(13)
    baseline = SiblingRestormer(**kwargs(), refinement_type="none")
    for refinement_type in ("transformer", "naf", "alcrb", "rerh", "lmrb", "haar"):
        torch.manual_seed(13)
        candidate = SiblingRestormer(**kwargs(), refinement_type=refinement_type)
        for name, parameter in baseline.named_parameters():
            if not name.startswith("refine_module."):
                assert torch.equal(parameter, dict(candidate.named_parameters())[name]), (refinement_type, name)


def test_single_final_clamp_for_rerh():
    model = SiblingRestormer(**kwargs(), refinement_type="rerh")
    image = torch.rand(1, 3, 32, 32)
    output = model(image)["restored"]
    assert float(output.min()) >= 0.0
    assert float(output.max()) <= 1.0
