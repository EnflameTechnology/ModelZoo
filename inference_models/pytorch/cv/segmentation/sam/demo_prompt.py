#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2022 Enflame. All Rights Reserved.
#
import os
import cv2
import time
import json
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor


def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(description='Onnx Model Fp32/Fp16 inference', add_help=add_help)
    parser.add_argument('--checkpoint',
                        default='./models/sam_vit_b_01ec64.pth',
                        type=str,
                        help='The path to the SAM model checkpoint.')
    parser.add_argument('--device',
                        default='gcu',
                        type=str,
                        help='model inference device')
    parser.add_argument('--image_path',
                        default='./images',
                        type=str,
                        help='image path')
    parser.add_argument('--ann_file',
                        default='./annotation.json',
                        type=str,
                        help='image path')
    parser.add_argument('--save_path',
                        default='./output',
                        type=str,
                        help='results save path')
    return parser


def show_mask(mask, ax):
    if mask is None:
        return
    color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    if coords is None:
        return
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def show_box(box, ax):
    if box is None:
        return
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))


def show_results(metas, save_path='./output', save_all=False):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    image = metas['image']
    image_name = '.'.join(metas['name'].split('.')[:-1])
    idx = str(metas['idx'])
    masks = metas['masks']
    w = image.shape[1]
    h = image.shape[0]
    if ('points' in metas) and (metas['points'] is not None):
        input_points = metas['points']
        input_labels = metas['labels']
    else:
        input_points = None
        input_labels = None
    if ('bboxes' in metas) and (metas['bboxes'] is not None):
        input_bboxes = metas['bboxes']
    else:
        input_bboxes = None
    dpi=100
    plt.figure(figsize=(w/dpi,h/dpi),dpi=dpi)
    plt.margins(0,0)
    plt.imshow(image)
    show_mask(masks[0][0], plt.gca())
    show_points(input_points, input_labels, plt.gca())
    show_box(input_bboxes, plt.gca())
    plt.axis('off')
    output_0 = os.path.join('./output', image_name + '_' + idx + '_0.png')
    plt.savefig(output_0, bbox_inches='tight',pad_inches=0.0)

    if save_all:
        plt.figure(figsize=(w/dpi,h/dpi),dpi=dpi)
        plt.margins(0,0)
        plt.imshow(image)
        show_mask(masks[0][1], plt.gca())
        show_points(input_points, input_labels, plt.gca())
        show_box(input_bboxes, plt.gca())
        plt.axis('off')
        output_1 = os.path.join('./output', image_name + '_' + idx + '_1.png')
        plt.savefig(output_1, bbox_inches='tight', pad_inches=0.0)

        plt.figure(figsize=(w/dpi,h/dpi),dpi=dpi)
        plt.margins(0,0)
        plt.imshow(image)
        show_mask(masks[0][2], plt.gca())
        show_points(input_points, input_labels, plt.gca())
        show_box(input_bboxes, plt.gca())
        plt.axis('off')
        output_2 = os.path.join('./output', image_name + '_' + idx + '_2.png')
        plt.savefig(output_2, bbox_inches='tight', pad_inches=0.0)

        plt.figure(figsize=(w/dpi,h/dpi),dpi=dpi)
        plt.margins(0,0)
        plt.imshow(image)
        show_mask(masks[0][3], plt.gca())
        show_points(input_points, input_labels, plt.gca())
        show_box(input_bboxes, plt.gca())
        plt.axis('off')
        output_3 = os.path.join('./output', image_name + '_' + idx + '_3.png')
        plt.savefig(output_3, bbox_inches='tight', pad_inches=0.0)


def device_synchronize_get_time(device):
    if device == "cuda" :
        torch.cuda.synchronize()
    elif device == "gcu":
        torch.gcu.synchronize()
    return time.time()


class SAMDetector(object):
    def __init__(self,
                 checkpoint,
                 device='cpu'):
        self.checkpoint = checkpoint
        self.device = device
        self.model_type = self.get_model_type()
        self.model = sam_model_registry[self.model_type](checkpoint=self.checkpoint)
        self.model.to(device=self.device)
        self.predictor = SamPredictor(self.model)

    def get_model_type(self):
        if 'vit_h' in self.checkpoint:
            model_type = 'vit_h'
        elif 'vit_l' in self.checkpoint:
            model_type = 'vit_l'
        elif 'vit_b' in self.checkpoint:
            model_type = 'vit_b'
        else:
            raise TypeError(f'Unknown model type from checkpoint: {model_file}')
        return model_type

    def forward(self, metas):
        self.predictor.set_image(metas['image'])
        point_coords = metas['points'] if 'points' in metas else None
        point_labels = metas['labels'] if 'labels' in metas else None
        box = metas['bboxes'] if 'bboxes' in metas else None
        mask_input = metas['mask_input'] if 'mask_input' in metas else None

        # multimask_output=False
        masks1, scores1, logits1 = self.predictor.predict(point_coords=point_coords,
                                                          point_labels=point_labels,
                                                          box=box,
                                                          mask_input=mask_input,
                                                          multimask_output=False)
        # multimask_output=True
        masks2, scores2, logits2 = self.predictor.predict(point_coords=point_coords,
                                                          point_labels=point_labels,
                                                          box=box,
                                                          mask_input=mask_input,
                                                          multimask_output=True)
        masks = np.concatenate([masks1[None, :, :, :], masks2[None, :, :, :]], axis=1)
        scores = np.concatenate([scores1[None, :], scores2[None, :]], axis=1)
        logits = np.concatenate([logits1[None, :, :, :], logits2[None, :, :, :]], axis=1)
        metas['masks'] = masks
        metas['iou_predictions'] = scores
        metas['low_res_masks'] = logits
        return metas


def inference(model_file,
              image_path,
              device='cpu',
              ann_file='./annotation.json',
              save_path='./output/prompt'):
    detector = SAMDetector(checkpoint=model_file, device=device)
    annotations = json.load(open(ann_file, 'r'))
    total_infer_time = 0.0
    total_images = 0
    total_prompts = 0
    for key in annotations:
        image_file = os.path.join(image_path, key)
        if not os.path.exists(image_file):
            print(f'can not find image file: {image_file}')
            continue
        image = cv2.imread(image_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        metas = annotations[key]
        results = []
        for idx, meta in enumerate(metas):
            points = np.array(meta['points']) if 'points' in meta else None
            labels = np.array(meta['labels']) if 'labels' in meta else None
            bboxes = np.array(meta['bboxes']) if 'bboxes' in meta else None
            mask_input = results[meta['mask']]['mask'] if 'mask' in meta else None
            meta = {
                'name': key,
                'idx': idx,
                'image': image,
                'points': points,
                'labels': labels,
                'bboxes': bboxes,
                'mask_input': mask_input
            }
            infer_start = device_synchronize_get_time(device)
            meta = detector.forward(meta)
            infer_end = device_synchronize_get_time(device)
            total_infer_time += infer_end - infer_start
            total_prompts += 1
            mask = meta['low_res_masks'][0, 1:, :, :]
            scores = meta['iou_predictions'][0, 1:]
            mask = mask[np.argmax(scores), :, :][None, :, :]
            results.append(mask)

            show_results(meta, save_path=save_path, save_all=False)
        total_images += 1

    infer_info = {
        'device': device,
        'model': model_file,
        'save_path': save_path,
        'image_number': total_images,
        'prompt_number': total_prompts,
        'avg_infer_time_per_prompt(ms)': round((total_infer_time / total_prompts) * 1000, 2),
    }
    print('=================== results ===================')
    print(json.dumps(infer_info, indent=4))


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    if args.device == 'gcu':
        import torch_gcu
    inference(model_file=args.checkpoint,
              image_path=args.image_path,
              device=args.device,
              ann_file=args.ann_file,
              save_path=args.save_path)

