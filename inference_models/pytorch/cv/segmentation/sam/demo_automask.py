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
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor


def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(description='Onnx Model Fp32/Fp16 inference', add_help=add_help)
    parser.add_argument('--checkpoint',
                        default='./models/sam_vit_b_01ec64.pth',
                        type=str,
                        help='The path to the SAM model checkpoint.')
    parser.add_argument('--device',
                        default='cpu',
                        type=str,
                        help='model inference device')
    parser.add_argument('--image_path',
                        default='./images',
                        type=str,
                        help='image path')
    parser.add_argument('--save_path',
                        default='./output',
                        type=str,
                        help='results save path')
    return parser


def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)


def device_synchronize_get_time(device):
    if device == "cuda" :
        torch.cuda.synchronize()
    elif device == "gcu":
        torch.gcu.synchronize()
    return time.time()


def inference(model_file,
              image_path,
              device='cpu',
              save_path='./output'):
    if 'vit_h' in model_file:
        model_type = 'vit_h'
    elif 'vit_l' in model_file:
        model_type = 'vit_l'
    elif 'vit_b' in model_file:
        model_type = 'vit_b'
    else:
        raise TypeError(f'Unknown model type from checkpoint: {model_file}')

    sam = sam_model_registry[model_type](checkpoint=model_file)
    sam.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(sam)

    if not os.path.exists(save_path):
            os.makedirs(save_path)

    total_infer_time = 0.0
    total_images = 0
    for image_name in os.listdir(image_path):
        if image_name.split('.')[-1].lower() not in ['jpg', 'jpeg', 'bmp', 'png']:
            continue
        image = cv2.imread(os.path.join(image_path, image_name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(20,20))
        plt.imshow(image)
        infer_start = device_synchronize_get_time(device)
        masks = mask_generator.generate(image)
        infer_end = device_synchronize_get_time(device)
        total_infer_time += infer_end - infer_start
        total_images += 1
        show_anns(masks)

        save_name = '.'.join(image_name.split('.')[:-1]) + '_out.jpg'
        output_name = os.path.join(save_path, save_name)
        plt.savefig(output_name, bbox_inches='tight',pad_inches=0.0)

    infer_info = {
        'device': device,
        'model': model_file,
        'save_path': save_path,
        'image_number': total_images,
        'avg_infer_time_per_image(ms)': round((total_infer_time / total_images) * 1000, 2),
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
              save_path=args.save_path)

