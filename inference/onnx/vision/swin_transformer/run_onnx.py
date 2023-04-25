#!/usr/bin/python3
#
# Copyright 2022 Enflame. All Rights Reserved.
#

import os
from collections import OrderedDict
import argparse
import onnxruntime
import numpy as np
import argparse

from common.img_preprocess import img_resize, img_center_crop
from common.logger import tops_logger, final_report
from common.utils import get_provider
from common.dataloader import DataLoader
from common.dataset import Dataset
from PIL import Image


def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(description='Swin Transformer ONNX Inference',
                                     add_help=add_help)
    parser.add_argument('--img_size',
                        default=224,
                        type=int,
                        help='model input image height')
    parser.add_argument('--model',
                        default='swin_tiny_patch4_window7_224-ms-op13-fp32-N.onnx',
                        help='onnx path')
    parser.add_argument("--dataset",
                        help="dataset path")
    parser.add_argument("--device",
                        choices=["gcu", "gpu", "cpu"],
                        default="gcu",
                        help="Which device will be used in inference, choices are ['gcu', 'gpu', 'cpu']. ")
    parser.add_argument('--batch_size',
                        default=1,
                        type=int,
                        help='batch size')
    parser.add_argument('--num_workers',
                        default=0,
                        type=int,
                        help='data loader workers number')
    return parser


def preprocess(img_file, args):
    image = Image.open(img_file).convert("RGB")
    input_size = (args.img_size, args.img_size)
    max_size = max(args.img_size, args.img_size)

    image = img_resize(image, 256, interpolation=Image.BICUBIC)
    image = img_center_crop(image, input_size)

    image_data = np.array(image, dtype='float32')
    IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
    norm_image_data = (image_data / 255 - IMAGENET_DEFAULT_MEAN) / IMAGENET_DEFAULT_STD

    norm_image_data = norm_image_data.reshape(args.img_size, args.img_size, 3).astype('float32')
    norm_image_data = np.array(norm_image_data).transpose(2, 0, 1)
    return norm_image_data

class ImageNet(Dataset):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        with open(os.path.join(args.dataset, 'val_map.txt'), "r") as file_path:
            self.val_map = file_path.readlines()

    def __getitem__(self, index):
        line = self.val_map[index]

        img_file, label = line.split(' ', -1)
        img_file = os.path.join(self.args.dataset, img_file)
        label = np.int32(int(label)) + 1
        input_data = preprocess(img_file, self.args)
        return {'input': input_data, 'label': label}

    def __len__(self):
        return len(self.val_map)

def arg_topk(array, k=5, axis=-1):
    topk_ind_unsort = np.argpartition(
        array, -k, axis=axis).take(indices=range(-k, 0), axis=axis)
    return topk_ind_unsort


def main(args):
    logger = tops_logger()
    provider = get_provider(args.device)
    session = onnxruntime.InferenceSession(args.model, providers=[provider])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    logger.info('Input Name:' + input_name)
    logger.info('Output Name:' + output_name)

    acc = 0
    acc5 = 0

    dataset = ImageNet(args)
    dataloader = DataLoader(dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers, drop_last=False)
    for idx, batch in enumerate(dataloader):
        label = batch['label']
        res = session.run([output_name], {input_name: batch['input']})
        res = res[0]

        indices = arg_topk(res)+1
        pred = np.argmax(res, axis=-1)+1
        acc += np.sum(pred == label)
        acc5 += (label[..., None] == indices).any(axis=-1).sum()
        logger.info('%d/%d ACC1=%f ACC5=%f' % (idx+1, len(dataloader), acc/((idx+1)*args.batch_size), acc5/((idx+1)*args.batch_size)))

    runtime_info = OrderedDict(
        [('model', args.model),
         ('dataset', args.dataset),
         ('batch_size', args.batch_size),
         ('device', args.device),
         ('acc1', acc / len(dataset)),
         ('acc5', acc5 / len(dataset))
         ])
    final_report(logger, runtime_info)

if __name__ == '__main__':
    args = get_args_parser().parse_args()
    main(args)
