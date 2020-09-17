import torch as t
import numpy as np
import torch.nn as nn

import six
from six import __init__
from torchvision.ops import nms
from torchvision.models import vgg16
from torchvision.ops import RoIPool

from config import opt


def tonumpy(data):
    if isinstance(data, np.ndarray):
        return data
    if isinstance(data, t.Tensor):
        return data.detach().cpu().numpy()


def totensor(data, cuda=True):
    if isinstance(data, np.ndarray):
        tensor = t.from_numpy(data)
    if isinstance(data, t.Tensor):
        tensor = data.detach()
    if cuda:
        tensor = tensor.cuda()
    return tensor


def scalar(data):
    if isinstance(data, np.ndarray):
        return data.reshape(1)[0]
    if isinstance(data, t.Tensor):
        return data.item()


def loc2bbox(src_bbox, loc):
    if src_bbox.shape[0] == 0:
        return np.zeros((0, 4), dtype=loc.dtype)
    src_bbox = src_bbox.astype(src_bbox.dtype, copy=False)
    src_height = src_bbox[:, 2] - src_bbox[:, 0]
    src_width = src_bbox[:, 3] - src_bbox[:, 1]
    src_ctr_y = src_bbox[:, 0] + 0.5 * src_height
    src_ctr_x = src_bbox[:, 1] + 0.5 * src_width

    dy = loc[:, 0::4]
    dx = loc[:, 1::4]
    dh = loc[:, 2::4]
    dw = loc[:, 3::4]

    ctr_y = dy * src_height[:, np.newaxis] + src_ctr_y[:, np.newaxis]
    ctr_x = dx * src_width[:, np.newaxis] + src_ctr_x[:, np.newaxis]
    
    h = np.exp(dh) * src_height[:, np.newaxis]
    w = np.exp(dw) * src_width[:, np.newaxis]
    print('h, w.shape: ', h.shape, w.shape)

    dst_bbox = np.zeros(loc.shape, dtype=loc.dtype)
    dst_bbox[:, 0::4] = ctr_y - 0.5 * h
    dst_bbox[:, 1::4] = ctr_x - 0.5 * w
    dst_bbox[:, 2::4] = ctr_y + 0.5 * h
    dst_bbox[:, 3::4] = ctr_x + 0.5 * w

    return dst_bbox


def bbox2loc(src_bbox, dst_bbox):
    """Encodes the source and the destination bounding boxes to "loc".
    Given bounding boxes, this function computes offsets and scales
    to match the source bounding boxes to the target bounding boxes.
    Mathematcially, given a bounding box whose center is
    :math:`(y, x) = p_y, p_x` and
    size :math:`p_h, p_w` and the target bounding box whose center is
    :math:`g_y, g_x` and size :math:`g_h, g_w`, the offsets and scales
    :math:`t_y, t_x, t_h, t_w` can be computed by the following formulas.
    * :math:`t_y = \\frac{(g_y - p_y)} {p_h}`
    * :math:`t_x = \\frac{(g_x - p_x)} {p_w}`
    * :math:`t_h = \\log(\\frac{g_h} {p_h})`
    * :math:`t_w = \\log(\\frac{g_w} {p_w})`
    The output is same type as the type of the inputs.
    The encoding formulas are used in works such as R-CNN [#]_.
    .. [#] Ross Girshick, Jeff Donahue, Trevor Darrell, Jitendra Malik. \
    Rich feature hierarchies for accurate object detection and semantic \
    segmentation. CVPR 2014.
    Args:
        src_bbox (array): An image coordinate array whose shape is
            :math:`(R, 4)`. :math:`R` is the number of bounding boxes.
            These coordinates are
            :math:`p_{ymin}, p_{xmin}, p_{ymax}, p_{xmax}`.
        dst_bbox (array): An image coordinate array whose shape is
            :math:`(R, 4)`.
            These coordinates are
            :math:`g_{ymin}, g_{xmin}, g_{ymax}, g_{xmax}`.
    Returns:
        array:
        Bounding box offsets and scales from :obj:`src_bbox` \
        to :obj:`dst_bbox`. \
        This has shape :math:`(R, 4)`.
        The second axis contains four values :math:`t_y, t_x, t_h, t_w`.
    """

    height = src_bbox[:, 2] - src_bbox[:, 0]
    width = src_bbox[:, 3] - src_bbox[:, 1]
    ctr_y = src_bbox[:, 0] + 0.5 * height
    ctr_x = src_bbox[:, 1] + 0.5 * width

    base_height = dst_bbox[:, 2] - dst_bbox[:, 0]
    base_width = dst_bbox[:, 3] - dst_bbox[:, 1]
    base_ctr_y = dst_bbox[:, 0] + 0.5 * base_height
    base_ctr_x = dst_bbox[:, 1] + 0.5 * base_width

    eps = np.finfo(height.dtype).eps
    height = np.maximum(height, eps)
    width = np.maximum(width, eps)

    dy = (base_ctr_y - ctr_y) / height
    dx = (base_ctr_x - ctr_x) / width
    dh = np.log(base_height / height)
    dw = np.log(base_width / width)

    loc = np.vstack((dy, dx, dh, dw)).transpose()
    return loc


def bbox_iou(bbox_a, bbox_b):
    """Calculate the Intersection of Unions (IoUs) between bounding boxes."""
    if bbox_a.shape[1] != 4 or bbox_b.shape[1] != 4:
        raise IndexError
    tl = np.maximum(bbox_a[:, None, :2], bbox_b[:, :2])
    br = np.maximum(bbox_a[:, None, 2:], bbox_b[:, 2:])

    area_i = np.prod(br - tl, axis=2) * (tl<br).all(axis=2)
    area_a = np.prod(bbox_a[:, 2:] - bbox_a[:, :2], axis=1)
    area_b = np.prod(bbox_b[:, 2:] - bbox_b[:, :2], axis=1)
    return area_i / (area_a[:, None] + area_b - area_i)


def generate_anchor_base(base_size=16, ratios=[0.5, 1, 2], anchor_scales=[8, 16, 32]):
    py = base_size / 2
    px = base_size / 2
    anchor_base = np.zeros((len(ratios) * len(anchor_scales), 4), dtype=np.float32)
    for i in six.moves.range(len(ratios)):
        for j in six.moves.range(len(anchor_scales)):
            h = base_size * anchor_scales[j] * np.sqrt(ratios[i])
            w = base_size * anchor_scales[j] * np.sqrt(1. / ratios[i])
            index = i * len(anchor_scales) + j
            anchor_base[index, 0] = py - h / 2
            anchor_base[index, 1] = py - w / 2
            anchor_base[index, 2] = py + h / 2
            anchor_base[index, 3] = py + w / 2
    return anchor_base


def normal_init(m, mean, stddev, truncated=False):
    """weight initalizer: truncated normal and random normal. """
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()


def _enumerate_shifted_anchor(anchor_base, feat_stride, height, width):
    """Enumerate all shifted anchors: (numpy)

    add A anchors (1, A, 4) to

    cell K shifts (K, 1, 4) to get

    shift anchors (K, A, 4)

    reshape to (K*A, 4) shifted anchors

    return (K*A, 4)
    """
    shift_y = np.arange(0, height * feat_stride, feat_stride)
    shift_x = np.arange(0, width * feat_stride, feat_stride)
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shift = np.stack((shift_y.ravel(), shift_x.ravel(), shift_y.ravel(), shift_x.ravel()), axis=1)

    A = anchor_base.shape[0]
    K = shift.shape[0]
    anchor = anchor_base.reshape((1, A, 4)) + shift.reshape((1, K, 4)).transpose((1, 0, 2))
    anchor = anchor.reshape((K * A, 4)).astype(np.float32)
    return anchor


def _enumerate_shifted_anchor_torch(anchor_base, feat_stride, height, width):
    """Enumerate all shifted anchors: (torch)

    add A anchors (1, A, 4) to

    cell K shifts (K, 1, 4) to get

    shift anchors (K, A, 4)

    reshape to (K*A, 4) shifted anchors

    return (K*A, 4)
    """
    shift_y = t.arange(0, height * feat_stride, feat_stride)
    shift_x = t.arange(0, width * feat_stride, feat_stride)
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shift = np.stack((shift_y.ravel(), shift_x.ravel(), shift_y.ravel(), shift_x.ravel()), axis=1)

    A = anchor_base.shape[0]
    K = shift.shape[0]
    print('<function> _enumerate_shifted_anchor_torch -> anchor_base.shape, shift.shape', anchor_base.shape, shift.shape)
    anchor = anchor_base.reshape((1, A, 4)) + shift.reshape((1, K, 4)).transpose((1, 0, 2))
    anchor = anchor.reshape((K * A, 4)).astype(np.float32)
    return anchor


class ProposalTargetCreator(object):
    """为2000个rois赋予ground truth，（调出128个赋予ground truth）【RoIHead】
    
    好了，到了最后一个需要解释的部分了，唉，这个博客感觉写了好久啊， 整整白天一天！ Proposal_TragetCreator的作用又是什么呢？简略点说那就是提供GroundTruth样本供ROISHeads网络进行自我训练的！那这个ROISHeads网络又是什么呢？就是接收ROIS对它进行n_class类别的预测以及最终目标检测位置的！也就是最终输出结果的网络啊，你说它重要不重要！最终输出结果的网络的性能的好坏完全取决于它，肯定重要呗！同样解释下它的流程：

    ProposalCreator产生2000个ROIS，但是这些ROIS并不都用于训练，经过本ProposalTargetCreator的筛选产生128个用于自身的训练，规则如下:

        1 ROIS和GroundTruth_bbox的IOU大于0.5,选取一些(比如说本实验的32个)作为正样本

        2 选取ROIS和GroundTruth_bbox的IOUS小于等于0的选取一些比如说选取128-32=96个作为负样本

        3然后分别对ROI_Headers进行训练

    补充：

        因为这些数据是要放入到整个大网络里进行训练的，比如说位置数据，所以要对其位置坐标进行数据增强处理(归一化处理)

        首先确定bbox.shape找出n_bbox的个数，然后将bbox和rois连接起来，确定需要的正样本的个数，调用bbox_iou进行IOU的计算，按行找到最大值，返回最大值对应的序号以及其真正的IOU，之后利用最大值的序号将那些挑出的最大值的label+1从0,n_fg_class-1 变到1,n_fg_class，同样的根据IOUS的最大值将正负样本找出来，如果找出的样本数目过多就随机丢掉一些，之后将正负样本序号连接起来，得到它们对应的真实的label，然后统一的将负样本的label全部置为0，这样筛选出来的样本的label就已经确定了，之后将sample_rois取出来，根据它和bbox的偏移量计算出loc,最后返回sample_rois,gt_roi_loc和gt_rois_label，完成任务使命！
    """
    def __init__(self, n_sample=128, pos_ratio=0.25, pos_iou_thresh=0.5, neg_iou_thresh_hi=0.5, neg_iou_thresh_lo=0.0):
        self.n_sample = n_sample
        self.pos_ratio = pos_ratio
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh_hi = neg_iou_thresh_hi
        self.neg_iou_thresh_lo = neg_iou_thresh_lo

    def __call__(self, roi, bbox, label, loc_normalize_mean=(0., 0., 0., 0.), loc_normalize_std=(0.1, 0.1, 0.2, 0.2)):
        n_bbox, _ = bbox.shape
        roi = np.concatenate((roi, bbox), axis=0)

        pos_roi_per_image = np.round(self.n_sample * self.pos_ratio)
        iou = bbox_iou(roi, bbox)
        gt_assignment = iou.argmax(axis=1)
        max_iou = iou.max(axis=1)
        gt_roi_label = label[gt_assignment] + 1

        pos_index = np.where(max_iou >= self.pos_iou_thresh)[0]
        pos_roi_per_this_image = int(min(pos_roi_per_image, pos_index.size))
        # print('pos_roi_per_this_image.type: ', type(pos_roi_per_this_image), type(pos_index[0]), pos_index)
        if pos_index.size > 0:
            pos_index = np.random.choice(
                pos_index, size=pos_roi_per_this_image, replace=False
            )
        neg_index = np.where((max_iou < self.neg_iou_thresh_hi) & (max_iou >= self.neg_iou_thresh_lo))[0]
        neg_roi_per_this_image = self.n_sample - pos_roi_per_this_image
        beg_roi_per_this_image = int(min(neg_roi_per_this_image, neg_index.size))


        if neg_index.size > 0:
            neg_index = np.random.choice(neg_index, size=neg_roi_per_this_image, replace=False)

        keep_index = np.append(pos_index, neg_index)            
        gt_roi_label = gt_roi_label[keep_index]
        gt_roi_label[pos_roi_per_this_image:] = 0
        sample_roi = roi[keep_index]

        gt_roi_loc = bbox2loc(sample_roi, bbox[gt_assignment[keep_index]])
        gt_roi_loc = ((gt_roi_loc - np.array(loc_normalize_mean, np.float32)) / np.array(loc_normalize_std, np.float32))
        
        return sample_roi, gt_roi_loc, gt_roi_label


def decom_vgg16():
    """
    vgg16_features:  Sequential(
            (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (1): ReLU(inplace=True)
            (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (3): ReLU(inplace=True)
            (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
            (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (6): ReLU(inplace=True)
            (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (8): ReLU(inplace=True)
            (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
            (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (11): ReLU(inplace=True)
            (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (13): ReLU(inplace=True)
            (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (15): ReLU(inplace=True)
            (16): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
            (17): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (18): ReLU(inplace=True)
            (19): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (20): ReLU(inplace=True)
            (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (22): ReLU(inplace=True)
            (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
            (24): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (25): ReLU(inplace=True)
            (26): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (27): ReLU(inplace=True)
            (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (29): ReLU(inplace=True)
            (30): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        )
    vgg16_classifier:  Sequential(
            (0): Linear(in_features=25088, out_features=4096, bias=True)
            (1): ReLU(inplace=True)
            (2): Dropout(p=0.5, inplace=False)
            (3): Linear(in_features=4096, out_features=4096, bias=True)
            (4): ReLU(inplace=True)
            (5): Dropout(p=0.5, inplace=False)
            (6): Linear(in_features=4096, out_features=1000, bias=True)
        )
    """
    model = vgg16()
    features = list(model.features)[:30]
    classifier = model.classifier
    classifier = list(classifier)
    del classifier[6]
    if not opt.vgg_use_drop:
        del classifier[5]
        del classifier[2]
    classifier = nn.Sequential(*classifier)

    for layer in features[:10]:
        for p in layer.parameters():
            p.requires_grad = False

    return nn.Sequential(*features), classifier


class AnchorTargetCreator:
    """用每张图的bbox的真是标签为所有任务分配ground truth/【RPN网络】
    
        之前我们直到_enumerate_shifted_anchor函数在一幅图上产生了20000多个anchor，而AnchorTargetCreator就是要从20000多个Anchor选出256个用于二分类和所有的位置回归！为预测值提供对应的真实值，选取的规则是：

        1.对于每一个Ground_truth bounding_box 从anchor中选取和它重叠度最高的一个anchor作为样本！

        2 从剩下的anchor中选取和Ground_truth bounding_box重叠度超过0.7的anchor作为样本，注意正样本的数目不能超过128

        3随机的从剩下的样本中选取和gt_bbox重叠度小于0.3的anchor作为负样本，正负样本之和为256

        PS:需要注意的是对于每一个anchor，gt_label要么为1,要么为0,所以这样实现二分类，而计算回归损失时，只有正样本计算损失，负样本不参与计算。

    过程：

        首先是读取图片的尺寸大小，然后用len(anchor)读取anchor的个数，一般对应20000个左右，之后调用_get_inside_index(anchor,img_H,img_W)来将那些超出图片范围的anchor全部去掉，mask只保留位于图片内部的，
        
        再调用self._create_label(inside_index,anchor,bbox)筛选出符合条件的正例128个负例128并给它们附上相应的label，最后调用bbox2loc将anchor和bbox进行求偏差当作回归计算的目标！

        -----

        展开来仔细看下_create_label 这个函数：

        首先初始化label,然后label.fill(-1)将所有标号全部置为-1,调用_clac_ious(anchor,bbox,inside_dex)产生argmax_ious,max_ious,gt_argmax_ious ， 
        
        之后进行判断，如果label[max_ious<self.neg_ious_thresh] = 0定义为负样本，而label[gt_argmax_ous]=1直接定义为正样本，同时label[max_ious>self.pos_iou_thresh]=1也定义为正样本，
        
        这里的定义规则其实gt_argmax_ious就是和gt_bbox重叠读最高的anchor，直接定义为正样本,而max_ious就是指的重叠度大于0.7的直接定义为正样本，而小于0.3的定义负样本，
        
        和开始讲的规则实际是一一对应的，程序接下来还有一个判断就是说如果选出来的label==1的个数大于pos_ratio*n_samples就是正样本如果按照这个规则选取多了的话，
        
        就调用np.random.choice(pos_index,size(len(pos_index)-n_pos),replace=False)就是总数不变随机丢弃掉一些正样本的意思！同样的方法如果负样本选择多了也随机丢弃掉一些，最后将序列argmax_ious,label返回！

        -----

        这里其实我写博客的时候有一句代码想不到合理的解释：就是loc = bbox2loc(anchor,bbox[argmax_ious]])  为什么anchor要和argmax_ious进行bbox2loc???到底是哪些和那些？一对一还是一对多啊？？ 
        
        后来我顿悟了，argmax_ious本来就是按顺序针对每一个anchor分别和IOUS进行交并比选取每一行的最大值产生的啊！
        
        argmax_ious只是列的序号，加上bbox就完成了bbox的选择工作，anchor自然要和对应最大的那个进行相交呗，还有一个问题就是为什么所有的anchor都要求bbox2loc?那可有20000个呢！！
        
        哈哈，记得我“答”那一行写的啥不，选出256用于二分类和所有的进行位置回归！是所有的啊，程序这里不就是很好的体现吗！
    """
    def __init__(self, n_sample=256, pos_iou_thresh=0.7, neg_iou_thresh=0.3, pos_ratio=0.5):
        self.n_sample = n_sample
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh = neg_iou_thresh
        self.pos_ratio = pos_ratio
    
    def __call__(self, bbox, anchor, img_size):
        img_H, img_W = img_size
        n_anchor = len(anchor)
        inside_index = _get_inside_index(anchor, img_H, img_W)
        anchor = anchor[inside_index] # inner anchor
        argmax_ious, label = self._create_label(inside_index, anchor, bbox)

        loc = bbox2loc(anchor, bbox[argmax_ious])

        label = _unmap(label, n_anchor, inside_index, fill=-1)
        loc = _unmap(loc, n_anchor, inside_index, fill=0)

        return loc, label

    def _create_label(self, inside_index, anchor, bbox):
        label = np.empty((len(inside_index), ), dtype=np.int32)
        label.fill(-1)

        argmax_ious, max_ious, gt_argmax_ious = self._calc_ious(anchor, bbox, inside_index)

        label[max_ious < self.neg_iou_thresh] = 0
        label[gt_argmax_ious] = 1
        label[max_ious >= self.pos_iou_thresh] = 1

        n_pos = int(self.pos_ratio * self.n_sample)
        pos_index = np.where(label==1)[0]
        if len(pos_index) > n_pos:
            disable_index = np.random.choice(pos_index, size=(len(pos_index) - n_pos), replace=False)
            label[disable_index] = -1
        
        n_neg = self.n_sample - np.sum(label == 1)
        neg_index = np.where(label == 0)[0]
        if len(neg_index) > n_neg:
            disable_index = np.random.choice(neg_index, size=(len(neg_index) - n_neg), replace=False)
            label[disable_index] = -1
        
        return argmax_ious, label

    def _calc_ious(self, anchor, bbox, inside_index):
        ious = bbox_iou(anchor, bbox)
        argmax_ious = ious.argmax(axis=1)
        max_ious = ious[np.arange(len(inside_index)), argmax_ious]
        gt_argmax_ious = ious.argmax(axis=0)
        gt_max_ious = ious[gt_argmax_ious, np.arange(ious.shape[1])]
        gt_argmax_ious = np.where(ious == gt_max_ious)[0]

        return argmax_ious, max_ious, gt_argmax_ious


def _unmap(data, count , index, fill=0):
    if len(data.shape) == 1:
        ret = np.empty((count, ), dtype=data.dtype)
        ret.fill(fill)
        ret[index] = data
    else:
        ret = np.empty((count, ) + data.shape[1:], dtype=data.dtype)
        ret.fill(fill)
        ret[index, :] = data
    return ret


def _get_inside_index(anchor, H, W):
    """Calc indices of anchors which are located completely inside of the image.
    
        找出位于图像内部的Anchor
    """
    index_inside = np.where(
        (anchor[:, 0] >= 0) &
        (anchor[:, 1] >= 0) &
        (anchor[:, 2] <= H) &
        (anchor[:, 3] <= W)
    )[0]
    return index_inside



class ProposalCreator:
    """输入上一张图的所有bbox,label的ground truth，输出的2000个roi作为ProposalTargetCreator的输入。【RPN网络】
    
        proposal的作用又是啥咧？ 其实proposalCreator做的就是生成ROIS的过程，而且整个过程只有前向计算没有反向传播，所以完全可以只用numpy和Tensor就把它计算出来咯！ 那具体的选取流程又是啥样的呢？
    1对于每张图片，利用FeatureMap,计算H/16*W/16*9大约20000个anchor属于前景的概率和其对应的位置参数，这个就是RPN网络正向作用的过程，没毛病，然后从中选取概率较大的12000张，
    利用位置回归参数，修正这12000个anchor的位置4 利用非极大值抑制，选出2000个ROIS！没错，整个流程读下来发现确实只有前向传播的过程

    流程:

        最开始初始化一些参数，比如nms_thresh=0.7,训练和测试选取不同的样本，min_size=16等等，果然代码一进来就针对训练和测试过程分别设置了不同的参数，
    然后rois = loc2bbox(anchor,loc)利用预测的修正值，对12000个anchor进行修正，

        之后利用numpy.clip(rois[:,slice(0,4,2)],0,img_size[0])函数将产生的rois的大小全部裁剪到图片范围以内！
    然后计算图片的高度和宽度，二者任何一个小于开始我们规定的min_size都直接mask掉！只保留剩下的Rois，然后对剩下的ROIs进行打分，
    对得到的分数进行合并然后进行排序，只保留属于前景的概率排序后的前12000/6000个（分别对应训练和测试时候的配置），之后再调用非极大值抑制函数，将重复的抑制掉，
    就可以将筛选后ROIS进行返回啦！ProposalCreator的函数的说明也结束了
    """
    def __init__(self, parent_model, nms_thresh=0.7, n_train_pre_nms=12000, n_train_post_nms=2000, n_test_pre_nms=6000, n_test_post_nms=300, min_size=16):
        self.parent_model = parent_model
        self.nms_thresh = nms_thresh
        self.n_train_pre_nms = n_train_pre_nms
        self.n_train_post_nms = n_train_post_nms
        self.n_test_pre_nms = n_test_pre_nms
        self.n_test_post_nms = n_test_post_nms
        self.min_size = min_size

    def __call__(self, loc, score, anchor, img_size, scale=1.):
        if self.parent_model.training:
            n_pre_nms = self.n_train_pre_nms
            n_post_nms = self.n_train_post_nms
        else:
            n_pre_nms = self.n_test_pre_nms
            n_post_nms = self.n_test_post_nms

        roi = loc2bbox(anchor, loc)
        print('<Class> ProposalCreator -> roi.shape: ', roi.shape)
        roi[:, slice(0, 4, 2)] = np.clip(roi[:, slice(0, 4, 2)], 0, img_size[0])
        roi[:, slice(1, 4, 2)] = np.clip(roi[:, slice(1, 4, 2)], 0, img_size[1])

        min_size = self.min_size * scale
        hs = roi[:, 2] - roi[:, 0]
        ws = roi[:, 3] - roi[:, 1]
        keep = np.where((hs >= min_size) & (ws >= min_size))[0]
        roi = roi[keep, :]
        score = score[keep]

        order = score.ravel().argsort()[::-1]
        if n_pre_nms > 0:
            order = order[:n_pre_nms]
        roi = roi[order, :]
        score = score[order]

        keep = nms(
            t.from_numpy(roi).cuda(),
            t.from_numpy(score).cuda(),
            self.nms_thresh
        )
        if n_post_nms > 0:
            keep = keep[:n_post_nms]
        roi = roi[keep.cpu().numpy()]
        return roi


class VGG16RoIHead(nn.Module):
    """FasterRCNN head for vgg16 based implementation."""
    def __init__(self, n_class, roi_size, spatial_scale, classifier):
        super(VGG16RoIHead, self).__init__()
        self.classifier = classifier
        self.cls_loc = nn.Linear(4096, n_class * 4)
        self.score = nn.Linear(4096, n_class)

        normal_init(self.cls_loc, 0, 0.001)
        normal_init(self.score, 0, 0.01)

        self.n_class = n_class
        self.roi_size = roi_size
        self.spatial_scale = spatial_scale
        self.roi = RoIPool((self.roi_size, self.roi_size), self.spatial_scale)

    def forward(self, x, rois, roi_indices):
        roi_indices = totensor(roi_indices).float()
        rois = totensor(rois).float()
        indices_and_rois = t.cat([roi_indices[:, None], rois], dim=1)
        xy_indices_and_rois = indices_and_rois[:, [0, 2, 1, 4, 3]]
        indices_and_rois = xy_indices_and_rois.contiguous()

        pool = self.roi(x, indices_and_rois)
        pool = pool.view(pool.size(0), -1)
        fc7 = self.classifier(pool)
        roi_cls_locs = self.cls_loc(fc7)
        roi_scores = self.score(fc7)
        return roi_cls_locs, roi_scores


if __name__ == '__main__':
    anchor_base = generate_anchor_base()
    print(anchor_base)