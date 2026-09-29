"""
Unit Tests for Depth Anything V2 Fine-Tuning Pipeline
Tests losses, datasets, forward/backward passes, and training integration.
"""

import os
import sys

# Ensure root workspace is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
import torch.nn.functional as F

from train.losses import (
    align_scale_shift,
    ScaleShiftInvariantLoss,
    MultiScaleGradientLoss,
    PearsonCorrelationLoss,
    CombinedAerialDepthLoss
)
from train.dataset import PotsdamTileDataset, GAMUSTileDataset, UnifiedAerialDepthDataset
from train.train_depth_anything import setup_model


def test_align_scale_shift():
    print("[TEST] test_align_scale_shift...")
    B, H, W = 2, 32, 32
    # Non-trivial linear affine relation
    pred = torch.linspace(1.0, 10.0, steps=B * H * W).reshape(B, H, W)
    target = 3.5 * pred + 4.2
    mask = torch.ones(B, H, W, dtype=torch.bool)

    aligned = align_scale_shift(pred, target, mask)
    if aligned.ndim == 4:
        aligned = aligned.squeeze(1)
    assert torch.allclose(aligned, target, atol=1e-3)

    # Constant fallback test
    c_target = torch.ones(B, H, W) * 10.0
    c_pred = torch.ones(B, H, W) * 2.0
    c_aligned = align_scale_shift(c_pred, c_target, mask)
    if c_aligned.ndim == 4:
        c_aligned = c_aligned.squeeze(1)
    assert torch.allclose(c_aligned, c_target, atol=1e-3)

    print("  [PASS] test_align_scale_shift passed.")


def test_losses_and_gradients():
    print("[TEST] test_losses_and_gradients...")
    B, H, W = 2, 64, 64
    pred = torch.randn(B, H, W, requires_grad=True)
    target = torch.randn(B, H, W) + 15.0
    mask = torch.ones(B, H, W, dtype=torch.bool)

    loss_fn = CombinedAerialDepthLoss(w_ssil=1.0, w_grad=0.5, w_corr=0.2)
    out = loss_fn(pred, target, mask)

    assert "loss" in out
    assert "loss_ssil" in out
    assert "loss_grad" in out
    assert "loss_corr" in out
    assert torch.isfinite(out["loss"]).item()

    out["loss"].backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all().item()
    print("  [PASS] test_losses_and_gradients passed.")


def test_potsdam_dataset_crops():
    print("[TEST] test_potsdam_dataset_crops...")
    ds = PotsdamTileDataset(
        potsdam_dir="data/potsdam",
        tile_ids=["02_10"],
        crop_size=128,
        samples_per_epoch=2,
        is_train=True
    )
    sample = ds[0]
    assert sample["pixel_values"].shape == (3, 128, 128)
    assert sample["depth"].shape == (128, 128)
    assert sample["valid_mask"].shape == (128, 128)
    assert sample["source"] == "potsdam"
    assert sample["depth"].min() >= 0.0
    print("  [PASS] test_potsdam_dataset_crops passed.")


def test_gamus_dataset_crops():
    print("[TEST] test_gamus_dataset_crops...")
    ds = GAMUSTileDataset(
        split="test",
        tile_ids=["DC_03_26"],
        crop_size=128,
        samples_per_epoch=2,
        is_train=False
    )
    sample = ds[0]
    assert sample["pixel_values"].shape == (3, 128, 128)
    assert sample["depth"].shape == (128, 128)
    assert sample["valid_mask"].shape == (128, 128)
    assert sample["source"] == "gamus"
    print("  [PASS] test_gamus_dataset_crops passed.")


def test_single_step_optimization():
    print("[TEST] test_single_step_optimization...")
    import argparse

    args = argparse.Namespace(
        pretrained_model="depth-anything/Depth-Anything-V2-Small-hf",
        freeze_backbone=True,
        unfreeze_backbone=False,
        use_lora=False,
        lora_r=16
    )
    device = torch.device("cpu")
    model, processor = setup_model(args, device)

    # Verify frozen backbone and trainable head/neck
    assert not next(model.backbone.parameters()).requires_grad
    assert next(model.neck.parameters()).requires_grad
    assert next(model.head.parameters()).requires_grad

    dummy_x = torch.randn(1, 3, 256, 256)
    dummy_y = torch.randn(1, 256, 256).abs()
    dummy_mask = torch.ones(1, 256, 256, dtype=torch.bool)

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    loss_fn = CombinedAerialDepthLoss()

    # Forward
    outputs = model(pixel_values=dummy_x)
    pred = outputs.predicted_depth
    pred_interp = F.interpolate(
        pred.unsqueeze(1), size=(256, 256), mode="bilinear", align_corners=True
    ).squeeze(1)

    loss_dict = loss_fn(pred_interp, dummy_y, dummy_mask)
    loss = loss_dict["loss"]
    loss.backward()
    optimizer.step()

    assert torch.isfinite(loss).item()
    print("  [PASS] test_single_step_optimization passed.")


if __name__ == "__main__":
    print("=" * 65)
    print("Running Training Pipeline Unit Tests")
    print("=" * 65)
    test_align_scale_shift()
    test_losses_and_gradients()
    test_potsdam_dataset_crops()
    test_gamus_dataset_crops()
    test_single_step_optimization()
    print("\nALL TRAINING PIPELINE UNIT TESTS PASSED SUCCESSFULLY!")
