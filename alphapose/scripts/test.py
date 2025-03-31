"""Validation script."""
import argparse
import json
import os
import numpy as np
import torch
from tqdm import tqdm
import sys

from alphapose.models import builder
from alphapose.utils.config import update_config
from alphapose.utils.metrics import evaluate_mAP
from alphapose.utils.transforms import (flip, flip_heatmap,
                                        get_func_heatmap_to_coord)
from alphapose.utils.metrics import DataLogger, calc_accuracy, calc_integral_accuracy


parser = argparse.ArgumentParser(description='AlphaPose Validate')
parser.add_argument('--cfg',
                    help='experiment configure file name',
                    required=True,
                    type=str)
parser.add_argument('--checkpoint',
                    help='checkpoint file name',
                    required=True,
                    type=str)
parser.add_argument('--gpus',
                    help='gpus',
                    default='0',
                    type=str)
parser.add_argument('--batch',
                    help='validation batch size',
                    default=32,
                    type=int)
parser.add_argument('--num_workers',
                    help='validation dataloader number of workers',
                    default=20,
                    type=int)
parser.add_argument('--flip-test',
                    default=False,
                    dest='flip_test',
                    help='flip test',
                    action='store_true')
parser.add_argument('--detector', dest='detector',
                    help='detector name', default="yolo")
parser.add_argument('--oks-nms',
                    default=False,
                    dest='oks_nms',
                    help='use oks nms',
                    action='store_true')
parser.add_argument('--ppose-nms',
                    default=False,
                    dest='ppose_nms',
                    help='use pPose nms, recommended',
                    action='store_true')

opt = parser.parse_args()
cfg = update_config(opt.cfg)

gpus = [int(i) for i in opt.gpus.split(',')]
opt.gpus = [gpus[0]]
opt.device = torch.device("cuda:" + str(opt.gpus[0]) if opt.gpus[0] >= 0 else "cpu")


def test_gt(m, opt, cfg, heatmap_to_coord, criterion, batch_size=20, num_workers=20):
    gt_val_dataset = builder.build_dataset(cfg.DATASET.TEST, preset_cfg=cfg.DATA_PRESET, train=False)
    gt_val_loader = torch.utils.data.DataLoader(
        gt_val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)

    loss_logger = DataLogger()
    acc_logger = DataLogger()

    m.eval()
    norm_type = cfg.LOSS.get('NORM_TYPE', None)

    with torch.no_grad():
        for inps, labels, label_masks, img_ids, bboxes in tqdm(gt_val_loader, dynamic_ncols=True):
            if isinstance(inps, list):
                inps = [inp.cuda() for inp in inps]
            else:
                inps = inps.cuda()

            if isinstance(labels, list):
                labels = [label.cuda() for label in labels]
                label_masks = [label_mask.cuda() for label_mask in label_masks]
            else:
                labels = labels.cuda()
                label_masks = label_masks.cuda()

            output = m(inps)

            if cfg.LOSS.get('TYPE') == 'MSELoss':
                loss = 0.5 * criterion(output.mul(label_masks), labels.mul(label_masks))
                acc = calc_accuracy(output.mul(label_masks), labels.mul(label_masks))
            elif cfg.LOSS.get('TYPE') == 'Combined':
                if output.size()[1] == 68:
                    face_hand_num = 42
                else:
                    face_hand_num = 110

                output_body_foot = output[:, :-face_hand_num, :, :]
                output_face_hand = output[:, -face_hand_num:, :, :]

                label_masks_body_foot = label_masks[0]
                label_masks_face_hand = label_masks[1]

                labels_body_foot = labels[0]
                labels_face_hand = labels[1]

                loss_body_foot = 0.5 * criterion[0](output_body_foot.mul(label_masks_body_foot), labels_body_foot.mul(label_masks_body_foot))
                acc_body_foot = calc_accuracy(output_body_foot.mul(label_masks_body_foot), labels_body_foot.mul(label_masks_body_foot))

                loss_face_hand = criterion[1](output_face_hand, labels_face_hand, label_masks_face_hand)
                acc_face_hand = calc_integral_accuracy(output_face_hand, labels_face_hand, label_masks_face_hand, output_3d=False, norm_type=norm_type)

                loss = loss_body_foot + loss_face_hand
                acc = acc_body_foot * (output_body_foot.shape[1] / output.shape[1]) + acc_face_hand * (output_face_hand.shape[1] / output.shape[1])
            else:
                loss = criterion(output, labels, label_masks)
                acc = calc_integral_accuracy(output, labels, label_masks, output_3d=False, norm_type=norm_type)

            batch_size = inps[0].size(0) if isinstance(inps, list) else inps.size(0)
            loss_logger.update(loss.item(), batch_size)
            acc_logger.update(acc, batch_size)

    return loss_logger.avg, acc_logger.avg



if __name__ == "__main__":
    
    # TODO get loss
    combined_loss = (cfg.LOSS.get('TYPE') == 'Combined')
    if combined_loss:
        criterion1 = builder.build_loss(cfg.LOSS.LOSS_1).cuda()
        criterion2 = builder.build_loss(cfg.LOSS.LOSS_2).cuda()
        criterion = [criterion1, criterion2]
    else:
        criterion = builder.build_loss(cfg.LOSS).cuda()
        
    m = builder.build_sppe(cfg.MODEL, preset_cfg=cfg.DATA_PRESET)

    print(f'Loading model from {opt.checkpoint}...')
    m.load_state_dict(torch.load(opt.checkpoint))

    m = torch.nn.DataParallel(m, device_ids=gpus).cuda()
    heatmap_to_coord = get_func_heatmap_to_coord(cfg)

    with torch.no_grad():
        gt_test_loss, gt_test_acc = test_gt(m, cfg, heatmap_to_coord, criterion, opt.batch, opt.num_workers)
        #detbox_AP = validate(m, heatmap_to_coord, opt.batch, opt.num_workers)
    print(f"GT Test Loss: {gt_test_loss}, Accuracy: {gt_test_acc}")
    #print('##### gt box: {} mAP | det box: {} mAP #####'.format(gt_AP, detbox_AP))