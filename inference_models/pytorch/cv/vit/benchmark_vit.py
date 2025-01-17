#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2024 Enflame. All Rights Reserved.
#
import torch
from torchvision import transforms
from transformers import ViTImageProcessor, ViTForImageClassification
from torchvision.datasets import ImageNet
from torch.utils.data import DataLoader
import argparse
import os
import json
import time

def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(description="ViT demo", add_help=add_help)

    parser.add_argument(
        "--model_dir",
        required=True,
        type=str,
        help="Specify the model_dir of vit pretrained_model, such as: ./models--google--vit-large-patch16-384")

    parser.add_argument(
        "--device",
        default="gcu",
        choices=["cpu", "cuda", "gcu"],
        help="Specify the device for testing (cpu, gpu, gcu)")

    parser.add_argument(
        '--data_path',
        required=True,
        type=str, help='path to ImageNet val')

    parser.add_argument(
        "--batch_size",
        default=256,
        type=int,
        help="Times of inference execution needed for preheating/warmup",
    )

    parser.add_argument(
        "--warmup_count",
        default=10,
        type=int,
        help="The number of batches needed for preheating/warmup",
    )

    parser.add_argument(
        "--eval_count",
        default=float("INF"),
        type=int,
        help="The number of batches used to evaluate performance and accuracy, where float('INF') represents traversing the entire dataset",
    )

    parser.add_argument(
        "--output_dir",
        default='./benchmark/vit-torch/',
        type=str,
        help="Specify the output_dir of vit benchmark-test report")

    return parser

def device_synchronize_get_time(device):
    if device == "cuda" :
        torch.cuda.synchronize()
    elif device == "gcu":
        torch.gcu.synchronize()
    return time.time()

def test_benchmark_vit(
    model_dir,
    dataset_path,
    device='gcu',
    batch_size=256,
    warmup_count=20,
    eval_count=float("INF"),
    ):

    if device == "cuda":
        dtype=torch.float16
    elif device == "gcu":
        import torch_gcu
        dtype = torch.float16
    elif device == 'cpu':
        dtype=torch.float32

    imagenet_folder = dataset_path

    model = ViTForImageClassification.from_pretrained(model_dir)
    model = model.to(device)
    model = model.to(dtype)
    model.eval()

    vitImageProcessor = ViTImageProcessor.from_pretrained(model_dir)

    transform = transforms.Compose([
        transforms.Resize(vitImageProcessor.size['height']),
        transforms.CenterCrop(vitImageProcessor.size['height']),
        transforms.ToTensor(),
    ])

    val_dataset = ImageNet(root=imagenet_folder, split='val', transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Warm up the model
    print(f'warming up...')
    for i, (images, labels) in enumerate(val_loader):
        if i >= warmup_count:
            break
        with torch.inference_mode():
            images = images.to(device).to(dtype)
            inputs = vitImageProcessor(images=images, do_rescale=False, return_tensors="pt")
            inputs = inputs.to(device).to(dtype)
            outputs = model(**inputs)

    # Repeat inference
    correct = 0
    total = 0
    total_infer_time = 0

    num_batch = 0
    print(f'evaluating on dataset...')
    with torch.inference_mode():
        for images, labels in val_loader:
            if num_batch >= eval_count:
                break
            images = images.to(device).to(dtype)
            inputs = vitImageProcessor(images=images, do_rescale=False, return_tensors="pt")
            inputs = inputs.to(device).to(dtype)

            t_infer_start = device_synchronize_get_time(device)
            outputs = model(**inputs)
            t_infer_end = device_synchronize_get_time(device)

            total_infer_time += t_infer_end - t_infer_start

            logits = outputs.logits
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted.cpu() == labels).sum().item()
            accuracy = 100 * correct / total
            if(num_batch % 10 == 0):
                print(f'current accuracy: {accuracy}, total: {total}')
            num_batch = num_batch + 1

    accuracy = 100 * correct / total
    print(f'Accuracy on ImageNet-1000 validation set using ViT: {accuracy:.2f}%')
    avg_infer_time = total_infer_time / num_batch / batch_size
    avg_infer_time_per_batch = total_infer_time / num_batch
    print(f'Avg infer time per batch: {avg_infer_time_per_batch}')
    fps = 1 / avg_infer_time

    performance_info_dict = {
        'device': device,
        'pretrained model':model_dir,
        "data_path":dataset_path,
        'output_dir':args.output_dir,
        'warmup_count': warmup_count,
        'eval_count': num_batch,
        'batch_size': batch_size,
        'acc1':accuracy,
        'FPS': fps
    }

    info_dict = {'performance_info': performance_info_dict}

    output_json = "benchmark_vit_{}_{}.json".format(
        device, str(dtype).split('.')[1])
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    report_path = os.path.join(args.output_dir, output_json)

    with open(report_path, 'w') as f:
        json.dump(info_dict, f, indent=4)

    json_object = json.dumps(info_dict, indent=4)
    print(json_object)

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    print(f'args: {args}')

    test_benchmark_vit(
        model_dir=args.model_dir,
        device=args.device,
        dataset_path=args.data_path,
        batch_size=args.batch_size,
        warmup_count=args.warmup_count,
        eval_count=args.eval_count,
    )