#!/usr/bin/env python
# encoding: utf-8
#
# Copyright 2022 Enflame. All Rights Reserved.
#

"""
Validate dbnet mobilenetv3
"""
import os
import argparse
import numpy as np
import onnxruntime as ort

from common.utils import get_provider
from common.logger import tops_logger,final_report
from collections import OrderedDict

from data_preprocess import *
from post_process import DBPostProcess
from eval_metric import DetMetric

def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(description='Onnx Model Fp32/Fp16 inference',
                                     add_help=add_help)
    parser.add_argument('--model',
                        default='./dbnet-mv3-640x640-ppocr-op13-fp32-N.onnx',
                        help='onnx path')
    parser.add_argument('--dataset',
                        default='./data',
                        type=str,
                        help='dataset path')
    parser.add_argument('--device',
                        default='cpu',
                        help='gcu, gpu, cpu')
    parser.add_argument('--batch_size',
                        default=1,
                        type=int,
                        help='batch size')
    parser.add_argument('--input_height',
                        default=640,
                        type=int,
                        help='model input image height')
    parser.add_argument('--input_width',
                        default=640,
                        type=int,
                        help='model input image width')
    return parser

def preprocess(data, trans_img, trans_label):
    for trans in trans_img:
        data = trans(data)
    for trans in trans_label:
        data = trans(data)
    return data

def main(args):
    logger = tops_logger()
    provider = get_provider(args.device)
    session = ort.InferenceSession(args.model, providers=[provider])
    input_name = session.get_inputs()[0].name
    label_filename = os.path.join(args.dataset, 'test_icdar2015_label.txt')
    with open(label_filename, 'rb') as f:
        labels = f.readlines()
    transforms_img = [
                        DecodeImage('BGR', False),
                        DetResizeForTest(image_shape=[args.input_height, args.input_width]),
                        NormalizeImage(scale=1.0/255, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], order='hwc'),
                        ToCHWImage()]
    transforms_label = [DetLabelEncode()]
    post_process = DBPostProcess()
    eval_metric = DetMetric()
    for i, line in enumerate(labels):
        data_line = line.decode('utf-8')
        substr = data_line.strip('\n').split('\t')
        filename = substr[0]
        label = substr[1]
        img_path = os.path.join(args.dataset, filename)
        print(img_path)
        with open(img_path, 'rb') as f:
            img = f.read()
        data = {'img_path': img_path, 'label': label, 'image': img}
        data = preprocess(data, transforms_img, transforms_label)
        img = np.expand_dims(data['image'], axis=0)
        gt_polys = np.expand_dims(data['polys'], axis=0)
        ignore_tags = np.expand_dims(data['ignore_tags'], axis=0)
        ratio_shapes = np.expand_dims(data['shape'], axis=0)
        res = session.run(None, input_feed={input_name: img})
        post_res = post_process(res[0], [data['shape']])
        batch = [img, ratio_shapes, gt_polys, ignore_tags]
        eval_metric(post_res, batch)
    metric = eval_metric.get_metric()

    runtime_info = OrderedDict([
    ('model', args.model),
    ('dataset', args.dataset),
    ('batch_size', args.batch_size),
    ('device', args.device),
    ('precision', metric['precision']),
    ('recall', metric['recall']),
    ('hmean', metric['hmean'])
    ])

    final_report(logger, runtime_info)

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
