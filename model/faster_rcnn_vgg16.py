import torch as t
import torch.nn as nn
from torchvision.models import vgg16

from model.region_proposal_network import RegionProposalNetwork
from model.faster_rcnn import FasterRCNN
from model.utils import decom_vgg16
from model.utils import VGG16RoIHead


class FasterRCNNVGG16(FasterRCNN):
    feat_stride = 16

    def __init__(self, n_fg_class=20, ratios=[0.5, 1, 2], anchor_scales=[8, 16, 32]):
        extractor, classifier = decom_vgg16()
        rpn = RegionProposalNetwork(
            512, 512, ratios=ratios,
            anchor_scales=anchor_scales,
            feat_stride=self.feat_stride
        )
        head = VGG16RoIHead(
            n_class=n_fg_class + 1, 
            roi_size=7, 
            spatial_scale=(1. / self.feat_stride),
            classifier=classifier
        )
        super(FasterRCNNVGG16, self).__init__(extractor, rpn, head,)