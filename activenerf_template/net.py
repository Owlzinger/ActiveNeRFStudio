import torch
from torch import Tensor, nn, cat
import numpy as np

import torch.nn.functional as F

class NeRF(nn.Module):
    def __init__(
        self,
        D=8,
        W=256,
        input_ch=3,
        input_ch_views=3,
        output_ch=4,
        skips=[4],
        use_viewdirs=True,
    ):
        """ """
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs

        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)]
            + [
                nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W)
                for i in range(D - 1)
            ]
        )

        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W // 2)])

        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])

        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W // 2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, x):
        input_pts, input_views = torch.split(
            x, [self.input_ch, self.input_ch_views], dim=-1
        )
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)

            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        return outputs


class activeNeRF(nn.Module):
    def __init__(
        self,
        D=8,
        W=256,
        input_ch=3,
        input_ch_views=3,
        output_ch=4,
        skips=[4],
        use_viewdirs=True,
        beta_min=0.0,
    ):
        """ """
        super(activeNeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        self.beta_min = beta_min
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)]
            + [
                nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W)
                for i in range(D - 1)
            ]
        )

        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W // 2)])

        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])

        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.uncertainty_linear = nn.Linear(W, 1)
            # self.act_uncertainty = nn.ReLU()
            self.act_uncertainty = nn.Softplus()
            self.rgb_linear = nn.Linear(W // 2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, x):
        input_pts, input_views = torch.split(
            x, [self.input_ch, self.input_ch_views], dim=-1
        )
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            uncert = self.act_uncertainty(self.uncertainty_linear(h)) + self.beta_min
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)

            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha, uncert], -1)
        else:
            outputs = self.output_linear(h)

        return outputs


if __name__ == "__main__":
    model = NeRF()
    model2 = activeNeRF()

    print(model)
    print(model2)
"""
NeRF(
  (pts_linears): ModuleList(
    (0): Linear(in_features=3, out_features=256, bias=True)
    (1-4): 4 x Linear(in_features=256, out_features=256, bias=True)
    (5): Linear(in_features=259, out_features=256, bias=True)
    (6-7): 2 x Linear(in_features=256, out_features=256, bias=True)
  )
  (views_linears): ModuleList(
    (0): Linear(in_features=259, out_features=128, bias=True)
  )
  (feature_linear): Linear(in_features=256, out_features=256, bias=True)
  (alpha_linear): Linear(in_features=256, out_features=1, bias=True)
  (rgb_linear): Linear(in_features=128, out_features=3, bias=True)
)
activeNeRF(
  (pts_linears): ModuleList(
    (0): Linear(in_features=3, out_features=256, bias=True)
    (1-4): 4 x Linear(in_features=256, out_features=256, bias=True)
    (5): Linear(in_features=259, out_features=256, bias=True)
    (6-7): 2 x Linear(in_features=256, out_features=256, bias=True)
  )
  (views_linears): ModuleList(
    (0): Linear(in_features=259, out_features=128, bias=True)
  )
  (feature_linear): Linear(in_features=256, out_features=256, bias=True)
  (alpha_linear): Linear(in_features=256, out_features=1, bias=True)
  (uncertainty_linear): Linear(in_features=256, out_features=1, bias=True)
  (act_uncertainty): Softplus(beta=1, threshold=20)
  (rgb_linear): Linear(in_features=128, out_features=3, bias=True)
)

"""