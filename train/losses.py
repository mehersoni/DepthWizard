"""
Loss functions for fine-tuning Depth Anything V2 on aerial remote sensing imagery.
Supports:
  1. Affine-Invariant Scale-and-Shift Loss (SSIL)
  2. Multi-Scale Edge-Aware Gradient Loss (preserving sharp building boundaries)
  3. Pearson Correlation Loss (directly maximizing linear correlation with LiDAR)
  4. Scale-Invariant Logarithmic (SILog) Loss
  5. Combined Aerial Depth Loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def align_scale_shift(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-6
) -> torch.Tensor:
    """
    Computes closed-form analytical scale (s) and shift (t) per sample
    such that sum_{valid} (s * pred + t - target)^2 is minimized:
        pred_aligned = s * pred + t
    Both pred and target are shape (B, 1, H, W) or (B, H, W).
    mask is a boolean tensor of the same shape indicating valid pixels.
    """
    if pred.ndim == 3:
        pred = pred.unsqueeze(1)
        target = target.unsqueeze(1)
        mask = mask.unsqueeze(1)

    B = pred.shape[0]
    aligned_list = []

    for b in range(B):
        p_b = pred[b]
        t_b = target[b]
        m_b = mask[b]

        p_valid = p_b[m_b]
        t_valid = t_b[m_b]

        n = p_valid.numel()
        if n < 10:
            # Degenerate case: fallback to raw prediction
            aligned_list.append(p_b)
            continue

        p_sum = torch.sum(p_valid)
        t_sum = torch.sum(t_valid)
        p2_sum = torch.sum(p_valid * p_valid)
        pt_sum = torch.sum(p_valid * t_valid)

        denom = n * p2_sum - p_sum * p_sum
        if torch.abs(denom) > eps:
            scale = (n * pt_sum - p_sum * t_sum) / denom
            shift = (t_sum - scale * p_sum) / n
        else:
            scale = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
            shift = t_sum / n

        aligned_list.append(scale * p_b + shift)

    aligned = torch.stack(aligned_list, dim=0)
    return aligned


class ScaleShiftInvariantLoss(nn.Module):
    """
    Scale-and-Shift Invariant L1 / Smooth L1 loss.
    Aligns prediction with target via analytical least-squares before computing L1 error.
    """
    def __init__(self, loss_type: str = "l1", eps: float = 1e-6):
        super().__init__()
        self.loss_type = loss_type
        self.eps = eps

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        if pred.ndim == 4 and pred.shape[1] == 1:
            pred = pred.squeeze(1)
        if target.ndim == 4 and target.shape[1] == 1:
            target = target.squeeze(1)
        if mask is not None and mask.ndim == 4 and mask.shape[1] == 1:
            mask = mask.squeeze(1)

        if mask is None:
            mask = torch.isfinite(target) & torch.isfinite(pred)
        else:
            mask = mask & torch.isfinite(target) & torch.isfinite(pred)

        if not mask.any():
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        aligned_pred = align_scale_shift(pred, target, mask, eps=self.eps)
        if aligned_pred.ndim == 4 and pred.ndim == 3:
            aligned_pred = aligned_pred.squeeze(1)

        if self.loss_type == "smooth_l1":
            diff = F.smooth_l1_loss(aligned_pred[mask], target[mask], reduction="mean")
        else:
            diff = torch.mean(torch.abs(aligned_pred[mask] - target[mask]))

        return diff


class MultiScaleGradientLoss(nn.Module):
    """
    Edge-preserving gradient loss across multiple spatial resolutions (1x, 0.5x, 0.25x).
    Forces the network to produce sharp step discontinuities at building rooftop edges.
    """
    def __init__(self, scales: int = 3):
        super().__init__()
        self.scales = scales

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        if pred.ndim == 3:
            pred = pred.unsqueeze(1)
            target = target.unsqueeze(1)
            if mask is not None:
                mask = mask.unsqueeze(1)

        if mask is None:
            mask = torch.isfinite(target) & torch.isfinite(pred)

        total_loss = torch.tensor(0.0, device=pred.device)
        curr_pred = pred
        curr_target = target
        curr_mask = mask.float()

        for s in range(self.scales):
            if s > 0:
                curr_pred = F.interpolate(curr_pred, scale_factor=0.5, mode="bilinear", align_corners=False)
                curr_target = F.interpolate(curr_target, scale_factor=0.5, mode="bilinear", align_corners=False)
                curr_mask = F.interpolate(curr_mask, scale_factor=0.5, mode="nearest")

            # Gradients in x and y
            diff_x = torch.abs((curr_pred[:, :, :, 1:] - curr_pred[:, :, :, :-1]) -
                               (curr_target[:, :, :, 1:] - curr_target[:, :, :, :-1]))
            mask_x = curr_mask[:, :, :, 1:] * curr_mask[:, :, :, :-1]

            diff_y = torch.abs((curr_pred[:, :, 1:, :] - curr_pred[:, :, :-1, :]) -
                               (curr_target[:, :, 1:, :] - curr_target[:, :, :-1, :]))
            mask_y = curr_mask[:, :, 1:, :] * curr_mask[:, :, :-1, :]

            denom_x = mask_x.sum().clamp(min=1.0)
            denom_y = mask_y.sum().clamp(min=1.0)

            loss_x = (diff_x * mask_x).sum() / denom_x
            loss_y = (diff_y * mask_y).sum() / denom_y

            total_loss = total_loss + (loss_x + loss_y)

        return total_loss / self.scales


class PearsonCorrelationLoss(nn.Module):
    """
    Differentiable Pearson Correlation loss:
        L_corr = 1 - r(pred, target)
    Directly maximizes the linear correlation coefficient between predicted depth
    and ground-truth LiDAR elevation.
    """
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        if pred.ndim == 3:
            pred = pred.unsqueeze(1)
            target = target.unsqueeze(1)
            if mask is not None:
                mask = mask.unsqueeze(1)

        if mask is None:
            mask = torch.isfinite(target) & torch.isfinite(pred)

        B = pred.shape[0]
        corrs = []

        for b in range(B):
            p = pred[b][mask[b]]
            t = target[b][mask[b]]

            if p.numel() < 10:
                corrs.append(torch.tensor(0.0, device=pred.device))
                continue

            p_centered = p - torch.mean(p)
            t_centered = t - torch.mean(t)

            cov = torch.sum(p_centered * t_centered)
            var_p = torch.sum(p_centered * p_centered)
            var_t = torch.sum(t_centered * t_centered)

            denom = torch.sqrt(var_p * var_t).clamp(min=self.eps)
            r = cov / denom
            corrs.append(r)

        mean_corr = torch.stack(corrs).mean()
        # Loss in [0, 2], 0 = perfect positive correlation
        return 1.0 - mean_corr


class SILogLoss(nn.Module):
    """
    Scale-Invariant Logarithmic (SILog) loss used in Monocular Depth Estimation benchmarks.
    L = sqrt(mean(d^2) - lambda * mean(d)^2) * alpha
    """
    def __init__(self, lambd: float = 0.85, alpha: float = 10.0, eps: float = 1e-6):
        super().__init__()
        self.lambd = lambd
        self.alpha = alpha
        self.eps = eps

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        if mask is None:
            mask = torch.isfinite(target) & torch.isfinite(pred) & (pred > 0) & (target > 0)
        else:
            mask = mask & torch.isfinite(target) & torch.isfinite(pred) & (pred > 0) & (target > 0)

        if not mask.any():
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        d = torch.log(pred[mask] + self.eps) - torch.log(target[mask] + self.eps)
        mean_d2 = torch.mean(d ** 2)
        mean_d = torch.mean(d)

        loss = torch.sqrt(torch.clamp(mean_d2 - self.lambd * (mean_d ** 2), min=0.0) + self.eps)
        return self.alpha * loss


class CombinedAerialDepthLoss(nn.Module):
    """
    Composite aerial loss designed for high-resolution building elevation:
      L = w_ssil * L_ssil + w_grad * L_grad + w_corr * L_corr
    """
    def __init__(
        self,
        w_ssil: float = 1.0,
        w_grad: float = 0.5,
        w_corr: float = 0.2,
        loss_type: str = "l1"
    ):
        super().__init__()
        self.w_ssil = w_ssil
        self.w_grad = w_grad
        self.w_corr = w_corr

        self.ssil = ScaleShiftInvariantLoss(loss_type=loss_type)
        self.grad_loss = MultiScaleGradientLoss(scales=3)
        self.corr_loss = PearsonCorrelationLoss()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor = None
    ) -> dict:
        l_ssil = self.ssil(pred, target, mask)
        l_grad = self.grad_loss(pred, target, mask)
        l_corr = self.corr_loss(pred, target, mask)

        total = self.w_ssil * l_ssil + self.w_grad * l_grad + self.w_corr * l_corr

        return {
            "loss": total,
            "loss_ssil": l_ssil.item(),
            "loss_grad": l_grad.item(),
            "loss_corr": l_corr.item(),
        }
