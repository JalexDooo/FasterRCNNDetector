import matplotlib.pyplot as plt
import matplotlib.image as Image

from data.dataset import VOC2007Detect
from model.faster_rcnn_vgg16 import FasterRCNNVGG16
from model.utils import totensor, tonumpy, scalar
from trainer.trainer import FasterRCNNTrainer
from utils.utils import eval_detection_voc

from config import opt

import ipdb
from torch.utils.data import DataLoader


def test_read_xml(**kwargs):
    opt._parse(kwargs)
    voc_xml_test = VOC2007Detect(opt.root_path, is_train=False)
    for img, bbox, label, scale, diff in voc_xml_test:
        print(img.shape, bbox, label, scale, diff)


def eval(dataloader, faster_rcnn, test_num=10000):
    pred_bboxes, pred_labels, pred_scores = list(), list(), list()
    gt_bboxes, gt_labels, gt_difficults = list(), list(), list()
    for ii, (imgs, sizes, gt_bboxes_, gt_labels_, gt_difficults_) in enumerate(dataloader):
        sizes = [sizes[0][0].item(), sizes[1][0].item()]
        pred_bboxes_, pred_labels_, pred_scores_ = faster_rcnn.predict(imgs, [sizes])
        gt_bboxes += list(gt_bboxes_.numpy())
        gt_labels += list(gt_labels_.numpy())
        gt_difficults += list(gt_difficults_.numpy())
        pred_bboxes += pred_bboxes_
        pred_labels += pred_labels_
        pred_scores += pred_scores_
        if ii == test_num: break

    result = eval_detection_voc(
        pred_bboxes, pred_labels, pred_scores,
        gt_bboxes, gt_labels, gt_difficults,
        use_07_metric=True)
    return result


def train(**kwargs):
    opt._parse(kwargs)

    dataset = VOC2007Detect(path=opt.root_path, is_train=True)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=opt.num_workers)

    test_dataset = VOC2007Detect(path=opt.root_path, is_train=False)

    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=opt.num_workers)
    print('training and testing data load finished.')

    print('\033[1;42m')
    print('---------Starting model initinalize.---------')

    faster_rcnn = FasterRCNNVGG16()
    print('model construct completed.')

    trainer = FasterRCNNTrainer(faster_rcnn).cuda()
    if opt.load_model_path:
        trainer.load(opt.load_model_path)
        print('load pretrained model from %s' % opt.load_model_path)

    print('len(dataloader): ', len(dataloader))

    print('\033[0m')
    print('---------Starting training stage.---------')
    best_map = 0
    lr_ = opt.lr
    for epoch in range(opt.max_epoch):
        trainer.reset_meters()
        for ii, (img, bbox_, label_, scale) in enumerate(dataloader):
            scale = scalar(scale)
            print('---------------------------------------------')
            print('Testing --> img.shape, bbox_.shape, label_.shape, scale: ', img.shape, bbox_.shape, label_.shape, scale)
            img, bbox, label = img.cuda().float(), bbox_.cuda(), label_.cuda()
            losses = trainer.train_step(img, bbox, label, scale)
            break

        eval_result = eval(test_dataloader, faster_rcnn, test_num=opt.test_num)
        lr_ = trainer.faster_rcnn.optimizer.param_groups[0]['lr']

        if eval_result['map'] > best_map:
            best_map = eval_result['map']
            best_path = trainer.save(best_map=best_map)


def other():
    print('\033[0;36m 字体有色，且有背景色')  # 无高亮


if __name__ == '__main__':
    import fire
    fire.Fire()
