#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2024 Enflame. All Rights Reserved.
#
import torch
from PIL import Image
import requests
import numpy as np
import argparse
from transformers import ViTImageProcessor, ViTForImageClassification

def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(description="ViT demo", add_help=add_help)

    parser.add_argument("--model_dir", required=True, type=str, help="path to ViT dir")

    parser.add_argument("--device", default='gcu', type=str, choices=['cpu', 'cuda', 'gcu'], help="Which device do you want to run the program on, CPU, GPU, or GCU?")

    parser.add_argument('--image_path',default=None, type=str, help='path to image, if leave it to be None, an image from web will be downloaded to be tested')

    return parser

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    print(f'args: {args}')

    if args.device == "cuda":
        dtype=torch.float16
    elif args.device == "gcu":
        import torch_gcu
        dtype = torch.float16
    elif args.device == 'cpu':
        dtype=torch.float32

    vitImageProcessor = ViTImageProcessor.from_pretrained(args.model_dir)
    model = ViTForImageClassification.from_pretrained(args.model_dir)
    model = model.to(args.device)
    model = model.to(dtype)
    model.eval()

    if(args.image_path is None):
        url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
        image = Image.open(requests.get(url, stream=True).raw)
    else:
        image = Image.open(args.image_path)

    image_tensor = torch.tensor(np.array(image))
    image_tensor = image_tensor.to(args.device).to(dtype)

    inputs = vitImageProcessor(images=image_tensor, return_tensors="pt")
    inputs = inputs.to(args.device).to(dtype)

    with torch.inference_mode():
        outputs = model(**inputs)

    logits = outputs.logits
    # model predicts one of the 1000 ImageNet classes
    predicted_class_idx = logits.argmax(-1).item()
    print("Predicted class:", model.config.id2label[predicted_class_idx])
