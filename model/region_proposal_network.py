import numpy as np
from torch.nn import functional as F
import torch as t
import torch.nn as nn

from model.utils import generate_anchor_base, normal_init, _enumerate_shifted_anchor_torch, _enumerate_shifted_anchor
from model.utils import ProposalCreator


class RegionProposalNetwork(nn.Module):
    """
    Params:
        self.anchor_base: generate 9 anchor base.

    """
    def __init__(self, in_channels=512, mid_channels=512, ratios=[0.5, 1, 2], anchor_scales=[8, 16, 32], feat_stride=16, proposal_creator_params=dict()):
        super(RegionProposalNetwork, self).__init__()
        self.anchor_base = generate_anchor_base(anchor_scales=anchor_scales, ratios=ratios)

        print('<Class __init__> RegionProposalNetwork -> self.anchor_base.shape: {}'.format(self.anchor_base.shape))

        self.feat_stride = feat_stride
        self.proposal_layer = ProposalCreator(self, **proposal_creator_params) ### Unknown function.
        n_anchor = self.anchor_base.shape[0]

        self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)
        self.score = nn.Conv2d(mid_channels, n_anchor * 2, 1, 1, 0) # 分类回归
        self.loc = nn.Conv2d(mid_channels, n_anchor * 4, 1, 1, 0) # 位置回归

        normal_init(self.conv1, 0, 0.01)
        normal_init(self.score, 0, 0.01)
        normal_init(self.loc, 0, 0.01)

    def forward(self, x, img_size, scale=1.):
        n, _, hh, ww = x.shape
        anchor = _enumerate_shifted_anchor(np.array(self.anchor_base), self.feat_stride, hh, ww)

        n_anchor = anchor.shape[0] // (hh * ww)
        h = F.relu(self.conv1(x))
        rpn_locs = self.loc(h)

        rpn_locs = rpn_locs.permute(0, 2, 3, 1).contiguous().view(n, -1, 4)
        rpn_scores = self.score(h)
        rpn_scores = rpn_scores.permute(0, 2, 3, 1).contiguous()
        rpn_softmax_scores = F.softmax(rpn_scores.view(n, hh, ww, n_anchor, 2), dim=4)
        rpn_fg_scores = rpn_softmax_scores[:, :, :, :, 1].contiguous()
        rpn_fg_scores = rpn_fg_scores.view(n, -1)
        rpn_scores = rpn_scores.view(n, -1, 2)

        rois = list()
        roi_indices = list()
        for i in range(n):
            roi = self.proposal_layer(
                rpn_locs[i].cpu().data.numpy(),
                rpn_fg_scores[i].cpu().data.numpy(),
                anchor, img_size,
                scale=scale
            )
            batch_index = i * np.ones((len(roi), ), dtype=np.int32)
            rois.append(roi)
            roi_indices.append(batch_index)

        rois = np.concatenate(rois, axis=0)
        roi_indices = np.concatenate(roi_indices, axis=0)
        return rpn_locs, rpn_scores, rois, roi_indices, anchor

