"""
Depth Anything V2 Fine-Tuning Pipeline for Remote Sensing & Aerial Orthophotography.
Fine-tunes on co-registered optical RGB imagery with LiDAR elevation (Potsdam LiDAR DSM & GAMUS AGL).

Key Features:
  - Freeze DINOv2 backbone (train 2.73M DPT neck/head params for maximum speed & stability)
  - Optional LoRA / PEFT adapter integration for transformer attention layers
  - Composite Aerial Loss: Scale-and-Shift Invariant L1 + Multi-Scale Gradient + Pearson Correlation
  - Mixed Multi-Source aerial dataset (ISPRS Potsdam + GAMUS)
  - Checkpoint saving in standard HuggingFace format and PyTorch state_dict
"""

import os
import sys
import json
import time
import argparse
from typing import Optional, Dict, Any, List
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

from train.losses import CombinedAerialDepthLoss, align_scale_shift
from train.dataset import UnifiedAerialDepthDataset, PotsdamTileDataset, GAMUSTileDataset
from evaluation.metrics import calculate_metrics

DEFAULT_MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"
DEFAULT_OUTPUT_DIR = "checkpoints/depth_anything_v2_aerial"


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Depth Anything V2 on Potsdam & GAMUS")
    parser.add_argument("--pretrained_model", type=str, default=DEFAULT_MODEL_ID,
                        help="HuggingFace model ID or local directory")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
                        help="Directory to save fine-tuned checkpoints and processor")
    parser.add_argument("--potsdam_dir", type=str, default="data/potsdam",
                        help="Path to ISPRS Potsdam data directory")
    parser.add_argument("--crop_size", type=int, default=518,
                        help="Spatial crop size (default 518, native Depth Anything V2 patch multiple)")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size per forward pass")
    parser.add_argument("--accum_steps", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of training epochs")
    parser.add_argument("--steps_per_epoch", type=int, default=20,
                        help="Number of optimization steps per epoch")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate for trainable parameters")
    parser.add_argument("--weight_decay", type=float, default=1e-2,
                        help="Weight decay for AdamW")
    parser.add_argument("--freeze_backbone", action="store_true", default=True,
                        help="Freeze DINOv2 backbone layers (default: True)")
    parser.add_argument("--unfreeze_backbone", action="store_true", default=False,
                        help="Unfreeze DINOv2 backbone for end-to-end training")
    parser.add_argument("--use_lora", action="store_true", default=False,
                        help="Apply LoRA to backbone attention layers via PEFT")
    parser.add_argument("--lora_r", type=int, default=16,
                        help="LoRA rank")
    parser.add_argument("--w_ssil", type=float, default=1.0,
                        help="Weight for Scale-and-Shift Invariant Loss")
    parser.add_argument("--w_grad", type=float, default=0.5,
                        help="Weight for Multi-scale Gradient Loss")
    parser.add_argument("--w_corr", type=float, default=0.2,
                        help="Weight for Pearson Correlation Loss")
    parser.add_argument("--potsdam_ratio", type=float, default=0.5,
                        help="Ratio of Potsdam vs GAMUS samples in unified dataset")
    parser.add_argument("--device", type=str, default=None,
                        help="Device: 'cuda', 'cpu', or auto-detect")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="DataLoader worker count")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    return parser.parse_args()


def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_model(args, device: torch.device):
    """
    Loads Depth Anything V2, configures trainable parameters (backbone freezing or LoRA).
    """
    print(f"\n[ModelSetup] Loading base model: '{args.pretrained_model}'...")
    processor = AutoImageProcessor.from_pretrained(args.pretrained_model)
    model = AutoModelForDepthEstimation.from_pretrained(args.pretrained_model)

    freeze_backbone = args.freeze_backbone and not args.unfreeze_backbone

    if args.use_lora:
        print(f"[ModelSetup] Applying LoRA (r={args.lora_r}) to backbone attention layers...")
        try:
            from peft import LoraConfig, get_peft_model
            lora_config = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_r * 2,
                target_modules=["query", "value"],
                lora_dropout=0.05,
                bias="none",
                modules_to_save=["neck", "head"]
            )
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
        except Exception as e:
            print(f"[ModelSetup] Warning: Failed to apply LoRA: {e}. Falling back to standard head tuning.")
            args.use_lora = False

    if not args.use_lora:
        if freeze_backbone:
            print("[ModelSetup] Freezing DINOv2 backbone. Fine-tuning neck + head (2.73M params)...")
            for param in model.backbone.parameters():
                param.requires_grad = False
            for param in model.neck.parameters():
                param.requires_grad = True
            for param in model.head.parameters():
                param.requires_grad = True
        else:
            print("[ModelSetup] Full model fine-tuning enabled (all layers trainable)...")
            for param in model.parameters():
                param.requires_grad = True

    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[ModelSetup] Total params: {total_params:,} | Trainable params: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")

    return model, processor


def evaluate_split(model, dataset, device: torch.device, max_samples: int = 8) -> Dict[str, float]:
    """
    Evaluates model on held-out dataset crops, reporting MAE, RMSE, Pearson Correlation, and SSIL.
    """
    model.eval()
    all_corrs = []
    all_maes = []
    all_rmses = []

    eval_samples = min(len(dataset), max_samples)

    with torch.no_grad():
        for i in range(eval_samples):
            sample = dataset[i]
            x = sample["pixel_values"].unsqueeze(0).to(device)  # (1, 3, H, W)
            target = sample["depth"].unsqueeze(0).to(device)    # (1, H, W)
            mask = sample["valid_mask"].unsqueeze(0).to(device) # (1, H, W)

            target_h, target_w = target.shape[-2], target.shape[-1]
            outputs = model(pixel_values=x)
            pred = outputs.predicted_depth

            if pred.shape[-2:] != (target_h, target_w):
                pred = F.interpolate(
                    pred.unsqueeze(1),
                    size=(target_h, target_w),
                    mode="bilinear",
                    align_corners=True
                ).squeeze(1)

            aligned_pred = align_scale_shift(pred, target, mask)
            if aligned_pred.ndim == 4:
                aligned_pred = aligned_pred.squeeze(1)

            p_np = aligned_pred[0].cpu().numpy()
            t_np = target[0].cpu().numpy()
            m_np = mask[0].cpu().numpy()

            try:
                metrics = calculate_metrics(p_np, t_np, valid_mask=m_np)
                all_corrs.append(metrics["correlation"])
                all_maes.append(metrics["mae"])
                all_rmses.append(metrics["rmse"])
            except Exception:
                continue

    if not all_corrs:
        return {"correlation": 0.0, "mae": 0.0, "rmse": 0.0}

    return {
        "correlation": float(np.mean(all_corrs)),
        "mae": float(np.mean(all_maes)),
        "rmse": float(np.mean(all_rmses))
    }


def train(args):
    set_seed(args.seed)

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print("=" * 70)
    print("DepthWizard - Depth Anything V2 Aerial Fine-Tuning Pipeline")
    print(f"Target Device: {device} | OS: {sys.platform}")
    print(f"Epochs: {args.epochs} | Steps/Epoch: {args.steps_per_epoch} | Batch Size: {args.batch_size} (accum: {args.accum_steps})")
    print(f"Crop Size: {args.crop_size}x{args.crop_size} | LR: {args.lr}")
    print("=" * 70)

    # 1. Datasets
    print("\n[Data] Initializing Unified Aerial Training Dataset (Potsdam + GAMUS)...")
    train_dataset = UnifiedAerialDepthDataset(
        potsdam_dir=args.potsdam_dir,
        crop_size=args.crop_size,
        samples_per_epoch=args.steps_per_epoch * args.batch_size * args.accum_steps,
        is_train=True,
        potsdam_ratio=args.potsdam_ratio,
        seed=args.seed
    )

    print("[Data] Initializing Held-out Validation Datasets...")
    val_potsdam_ds = PotsdamTileDataset(
        potsdam_dir=args.potsdam_dir,
        tile_ids=["02_12"],
        crop_size=args.crop_size,
        samples_per_epoch=8,
        is_train=False,
        seed=args.seed + 100
    )
    val_gamus_ds = GAMUSTileDataset(
        split="test",
        tile_ids=["DC_03_26"],
        crop_size=args.crop_size,
        samples_per_epoch=8,
        is_train=False,
        seed=args.seed + 101
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True
    )

    # 2. Model & Processor
    model, processor = setup_model(args, device)

    # 3. Loss & Optimizer
    loss_fn = CombinedAerialDepthLoss(
        w_ssil=args.w_ssil,
        w_grad=args.w_grad,
        w_corr=args.w_corr
    ).to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    total_training_steps = args.epochs * args.steps_per_epoch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_training_steps,
        eta_min=args.lr * 0.05
    )

    # 4. Zero-shot Baseline Validation
    print("\n[Baseline Evaluation] Measuring zero-shot frozen baseline metrics before fine-tuning...")
    val_potsdam_init = evaluate_split(model, val_potsdam_ds, device, max_samples=4)
    val_gamus_init = evaluate_split(model, val_gamus_ds, device, max_samples=4)
    print(f"  Potsdam Held-out (Tile 02_12): Corr = {val_potsdam_init['correlation']:.4f} | MAE = {val_potsdam_init['mae']:.2f} m | RMSE = {val_potsdam_init['rmse']:.2f} m")
    print(f"  GAMUS Held-out (Tile DC_03_26): Corr = {val_gamus_init['correlation']:.4f} | MAE = {val_gamus_init['mae']:.2f} m | RMSE = {val_gamus_init['rmse']:.2f} m")

    # 5. Training Loop
    history = {
        "epochs": [],
        "train_loss": [],
        "val_potsdam_corr": [val_potsdam_init["correlation"]],
        "val_gamus_corr": [val_gamus_init["correlation"]],
    }

    best_val_corr = (val_potsdam_init["correlation"] + val_gamus_init["correlation"]) / 2.0
    os.makedirs(args.output_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print("Beginning Fine-Tuning Execution...")
    print("=" * 70)

    global_step = 0
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses = []
        optimizer.zero_grad()

        epoch_start = time.time()
        step_in_epoch = 0

        for batch_idx, batch in enumerate(train_loader):
            if step_in_epoch >= args.steps_per_epoch:
                break

            pixel_values = batch["pixel_values"].to(device)
            target = batch["depth"].to(device)
            mask = batch["valid_mask"].to(device)

            target_h, target_w = target.shape[-2], target.shape[-1]

            # Forward pass
            outputs = model(pixel_values=pixel_values)
            pred = outputs.predicted_depth

            if pred.shape[-2:] != (target_h, target_w):
                pred = F.interpolate(
                    pred.unsqueeze(1),
                    size=(target_h, target_w),
                    mode="bilinear",
                    align_corners=True
                ).squeeze(1)

            loss_dict = loss_fn(pred, target, mask)
            loss = loss_dict["loss"] / args.accum_steps

            # Backward pass
            loss.backward()

            if (batch_idx + 1) % args.accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1
                step_in_epoch += 1

                raw_loss = loss_dict["loss"].item()
                epoch_losses.append(raw_loss)

                if step_in_epoch % max(1, args.steps_per_epoch // 4) == 0 or step_in_epoch == args.steps_per_epoch:
                    lr_curr = scheduler.get_last_lr()[0]
                    print(
                        f"Epoch [{epoch:02d}/{args.epochs:02d}] Step [{step_in_epoch:02d}/{args.steps_per_epoch:02d}] "
                        f"Loss: {raw_loss:.4f} (SSIL: {loss_dict['loss_ssil']:.3f}, Grad: {loss_dict['loss_grad']:.3f}, Corr: {loss_dict['loss_corr']:.3f}) | LR: {lr_curr:.2e}"
                    )

        mean_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        epoch_dur = time.time() - epoch_start

        # Validation
        metrics_potsdam = evaluate_split(model, val_potsdam_ds, device, max_samples=4)
        metrics_gamus = evaluate_split(model, val_gamus_ds, device, max_samples=4)
        avg_corr = (metrics_potsdam["correlation"] + metrics_gamus["correlation"]) / 2.0

        print(f"\n--- Epoch {epoch:02d} Summary ({epoch_dur:.1f}s) ---")
        print(f"  Train Loss: {mean_loss:.4f}")
        print(f"  Val Potsdam Corr: {metrics_potsdam['correlation']:.4f} | MAE: {metrics_potsdam['mae']:.2f} m | RMSE: {metrics_potsdam['rmse']:.2f} m")
        print(f"  Val GAMUS Corr:   {metrics_gamus['correlation']:.4f} | MAE: {metrics_gamus['mae']:.2f} m | RMSE: {metrics_gamus['rmse']:.2f} m")
        print(f"  Combined Validation Pearson r: {avg_corr:.4f}")

        history["epochs"].append(epoch)
        history["train_loss"].append(mean_loss)
        history["val_potsdam_corr"].append(metrics_potsdam["correlation"])
        history["val_gamus_corr"].append(metrics_gamus["correlation"])

        # Checkpoint Saving
        if avg_corr >= best_val_corr or epoch == args.epochs:
            best_val_corr = max(best_val_corr, avg_corr)
            print(f"  >>> Saving best fine-tuned model checkpoint to '{args.output_dir}'...")

            # Save in HuggingFace format for direct drop-in replacement
            model.save_pretrained(args.output_dir)
            processor.save_pretrained(args.output_dir)

            # Save PyTorch state dict and training metadata
            torch_ckpt = os.path.join(args.output_dir, "depth_anything_v2_potsdam_gamus.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "history": history,
                "best_val_corr": best_val_corr,
                "args": vars(args)
            }, torch_ckpt)

            print(f"  >>> Saved HuggingFace model and PyTorch checkpoint '{torch_ckpt}'.")

    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"Training Complete! Total Time: {total_time:.1f}s ({total_time / 60:.2f} mins)")
    print(f"Final Best Combined Pearson Correlation r: {best_val_corr:.4f}")
    print(f"Fine-tuned weights saved to: {os.path.abspath(args.output_dir)}")
    print("=" * 70)

    # Save history json
    history_file = os.path.join(args.output_dir, "training_history.json")
    with open(history_file, "w") as f:
        json.dump(history, f, indent=2)

    return history


if __name__ == "__main__":
    args = parse_args()
    train(args)
