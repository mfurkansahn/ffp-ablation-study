import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Flow_Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, gen_flows, gt_flows):
        return torch.mean(torch.abs(gen_flows - gt_flows))


class Intensity_Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, gen_frames, gt_frames):
        return torch.mean(torch.abs((gen_frames - gt_frames) ** 2))


class Gradient_Loss(nn.Module):
    def __init__(self, channels):
        super().__init__()

        pos = torch.from_numpy(np.identity(channels, dtype=np.float32))
        neg = -1 * pos
        # Note: when doing conv2d, the channel order is different from tensorflow, so do permutation.
        self.filter_x = torch.stack((neg, pos)).unsqueeze(0).permute(3, 2, 0, 1).cuda()
        self.filter_y = torch.stack((pos.unsqueeze(0), neg.unsqueeze(0))).permute(3, 2, 0, 1).cuda()

    def forward(self, gen_frames, gt_frames):
        # Do padding to match the  result of the original tensorflow implementation
        gen_frames_x = nn.functional.pad(gen_frames, [0, 1, 0, 0])
        gen_frames_y = nn.functional.pad(gen_frames, [0, 0, 0, 1])
        gt_frames_x = nn.functional.pad(gt_frames, [0, 1, 0, 0])
        gt_frames_y = nn.functional.pad(gt_frames, [0, 0, 0, 1])

        gen_dx = torch.abs(nn.functional.conv2d(gen_frames_x, self.filter_x))
        gen_dy = torch.abs(nn.functional.conv2d(gen_frames_y, self.filter_y))
        gt_dx = torch.abs(nn.functional.conv2d(gt_frames_x, self.filter_x))
        gt_dy = torch.abs(nn.functional.conv2d(gt_frames_y, self.filter_y))

        grad_diff_x = torch.abs(gt_dx - gen_dx)
        grad_diff_y = torch.abs(gt_dy - gen_dy)

        return torch.mean(grad_diff_x + grad_diff_y)


class Adversarial_Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, fake_outputs):
        # TODO: compare with torch.nn.MSELoss ?
        return torch.mean((fake_outputs - 1) ** 2 / 2)


class Discriminate_Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, real_outputs, fake_outputs):
        return torch.mean((real_outputs - 1) ** 2 / 2) + torch.mean(fake_outputs ** 2 / 2)

class MotionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()

    def forward(self, pred_motion, gt_motion):
        return self.l1(pred_motion, gt_motion)

class TemporalConsistencyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, frame_t, frame_t_minus_1):
        # pred        : predicted frame_{t+1}
        # frame_t     : frame_t
        # frame_t-1   : frame_{t-1}

        # temporal difference
        diff_pred = pred - frame_t
        diff_gt   = frame_t - frame_t_minus_1

        return F.l1_loss(diff_pred, diff_gt)

class BezierTrajectoryLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, frame_t_minus_1, frame_t, frame_t_plus_1):
        """
        En basit ve stabil Bezier yaklaşımı:
        frame_t_plus_1 ≈ 2*frame_t - frame_t_minus_1
        """
        bezier_pred = 2 * frame_t - frame_t_minus_1
        return F.l1_loss(frame_t_plus_1, bezier_pred)


