#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2022 Enflame. All Rights Reserved.
#
import argparse
import math
import os
from collections import OrderedDict

import cv2
import numpy as np
import onnxruntime
from tqdm import tqdm

from common.dataloader import DataLoader
from common.dataset import Dataset
from common.logger import final_report, tops_logger
from common.utils import get_provider


def get_args_parser(add_help=True):
    """
    Arg Parser
    """
    parser = argparse.ArgumentParser(description='PPOCR Rec ONNX Inference',
                                     add_help=add_help)
    parser.add_argument('--model',
                        default='./model/crnn-resnet34-en-ppocr-op13-fp32-N.onnx',
                        type=str,
                        help='onnx model file')
    parser.add_argument('--device',
                        default='cpu',
                        type=str,
                        help='which device will be used, choices are [gcu, gpu, cpu]')
    parser.add_argument('--dataset',
                        default='./data',
                        type=str,
                        help='dataset path')
    parser.add_argument('--batch_size',
                        type=int,
                        default=32,
                        help='datalader batch size')
    parser.add_argument('--num_workers',
                        type=int,
                        default=0,
                        help='dataloader worker number')
    parser.add_argument('--image_shape',
                        default=(3, 32, 100),
                        type=int,
                        nargs='+',
                        help='model input image shape')
    parser.add_argument('--rec_char_dict_path',
                        default=None,
                        type=str,
                        help='words list file')
    return parser


class LabelParser(object):
    """
    Label Parser
    """
    def __init__(self,
                 character_dict_path=None,
                 use_space_char=True,
                 reverse=False,
                 is_lower=True,
                 is_remove_duplicate=True,
                 logger=None):
        self.character_dict_path = character_dict_path
        self.use_space_char = use_space_char
        self.is_lower = is_lower
        self.reverse = reverse
        self.is_remove_duplicate = is_remove_duplicate
        self.ignored_tokens = [0]
        self.logger = tops_logger() if logger is None else logger
        self.dict, self.character = self._parse_charaters()

    def _parse_charaters(self):
        """
        parse character dict
        """
        if self.character_dict_path is None:
            character_str = "0123456789abcdefghijklmnopqrstuvwxyz"
            dict_character = list(character_str)
        else:
            with open(self.character_dict_path, "rb") as fin:
                lines = fin.readlines()
                for line in lines:
                    line = line.decode('utf-8').strip("\n").strip("\r\n")
                    character_str.append(line)
            if self.use_space_char:
                character_str.append(" ")
            dict_character = list(character_str)
            if 'arabic' in self.character_dict_path:
                self.reverse = True
        if self.use_space_char:
            dict_character = ['blank'] + dict_character
        chars = {}
        for i, char in enumerate(dict_character):
            chars[char] = i
        return chars, dict_character

    def decode(self, preds):
        """
        convert pred label to character
        """
        result_list = []
        if isinstance(preds, tuple) or isinstance(preds, list):
            preds = preds[-1]
        text_index = preds.argmax(axis=2)
        text_prob = preds.max(axis=2)
        batch_size = len(text_index)
        for batch_idx in range(batch_size):
            selection = np.ones(len(text_index[batch_idx]), dtype=bool)
            if self.is_remove_duplicate:
                selection[1:] = text_index[batch_idx][1:] != text_index[batch_idx][:-1]
            for ignored_token in self.ignored_tokens:
                selection &= text_index[batch_idx] != ignored_token

            char_list = [
                self.character[text_id]
                for text_id in text_index[batch_idx][selection]
            ]
            if text_prob is not None:
                conf_list = text_prob[batch_idx][selection]
            else:
                conf_list = [1] * len(selection)
            if len(conf_list) == 0:
                conf_list = [0]
            text = ''.join(char_list)

            if self.reverse:  # for arabic rec
                text = self.pred_reverse(text)

            result_list.append((text, np.mean(conf_list).tolist()))
        return result_list

    def encode(self, text):
        """
        prprocess label text
        """
        if self.is_lower:
            text = text.lower()
        text_list = []
        for char in text:
            if char not in self.dict:
                continue
            text_list.append(char)
        text = ''.join(text_list)
        return text


class TextRecognizer(object):
    """
    CRNN Recognizer
    """
    def __init__(self,
                 model_file,
                 label_parser,
                 device='cpu',
                 compiled_batchsize=1,
                 image_shape=(3, 32, 100),
                 logger=None):
        self.model_file = model_file
        self.device = device
        self.compiled_batchsize = compiled_batchsize
        self.image_shape = image_shape
        self.label_parser = label_parser
        self.ignored_tokens = [0]
        self.is_remove_duplicate = True
        self.reverse = False
        self.logger = tops_logger() if logger is None else logger
        self.provider = get_provider(self.device)
        provider_options = {'compiled_batchsize': self.compiled_batchsize}
        self.session = onnxruntime.InferenceSession(self.model_file,
                                                    providers=[self.provider],
                                                    provider_options=[provider_options])
        self.input_names = [node.name for node in self.session.get_inputs()]
        self.logger.info('model input names: {}'.format(self.input_names))
        self.output_names = [node.name for node in self.session.get_outputs()]
        self.logger.info('model output names: {}'.format(self.output_names))

    def forward(self, data):
        """
        """
        input_feed = {name: data[name] for name in self.input_names}
        outputs = self.session.run(self.output_names, input_feed=input_feed)
        results = self.label_parser.decode(outputs)
        return results


class IIIKDataSet(Dataset):
    """
    IIIK Dataset
    """
    def __init__(self,
                 data_path,
                 label_parser,
                 image_shape=(3, 32, 100),
                 logger=None):
        self.data_path = data_path
        self.label_parser = label_parser
        self.image_shape = image_shape
        self.img_mode = 'BGR'
        self.channel_first = False
        self.logger = tops_logger() if logger is None else logger
        self.label_file = '{}/test_label.txt'.format(self.data_path)
        self.anns = open(self.label_file, 'r').readlines()

    def _decode_image(self, data):
        """
        read image
        """
        img = data['x']
        if isinstance(img, str):
            with open(img, 'rb') as f:
                img = f.read()
            img = np.frombuffer(img, dtype='uint8')
        img = cv2.imdecode(img, 1)
        if self.img_mode == 'GRAY':
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif self.img_mode == 'RGB':
            assert img.shape[2] == 3, 'invalid shape of image[%s]' % (img.shape)
            img = img[:, :, ::-1]
        if self.channel_first:
            img = img.transpose((2, 0, 1))
        data['x'] = img
        return data

    def _rec_resize_image(self, data, padding=True, interpolation=cv2.INTER_LINEAR):
        """
        resize and normalize image
        """
        img = data['x']
        imgC, imgH, imgW = self.image_shape
        h = img.shape[0]
        w = img.shape[1]
        if not padding:
            resized_image = cv2.resize(
                img, (imgW, imgH), interpolation=interpolation)
            resized_w = imgW
        else:
            ratio = w / float(h)
            if math.ceil(imgH * ratio) > imgW:
                resized_w = imgW
            else:
                resized_w = int(math.ceil(imgH * ratio))
            resized_image = cv2.resize(img, (resized_w, imgH))
        resized_image = resized_image.astype('float32')
        if self.image_shape[0] == 1:
            resized_image = resized_image / 255
            resized_image = resized_image[np.newaxis, :]
        else:
            resized_image = resized_image.transpose((2, 0, 1)) / 255
        resized_image -= 0.5
        resized_image /= 0.5
        padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
        padding_im[:, :, 0:resized_w] = resized_image
        valid_ratio = min(1.0, float(resized_w / imgW))
        data['x'] = padding_im
        data['valid_ratio'] = valid_ratio
        return data

    def __getitem__(self, idx):
        """
        read data
        """
        ann = self.anns[idx]
        img_file, label = ann.strip('\n').split(' ')
        data = {
            'x': '{}/test/{}'.format(self.data_path, img_file),
            'label': self.label_parser.encode(label)
        }
        data = self._decode_image(data)
        data = self._rec_resize_image(data)
        return data

    def __len__(self):
        return len(self.anns)


if __name__ == '__main__':
    """
    """
    args = get_args_parser().parse_args()
    ignore_space=True
    logger = tops_logger()
    compiled_batchsize = max(int(args.batch_size / 6), 1)
    label_parser = LabelParser(character_dict_path=args.rec_char_dict_path,
                               logger=logger)
    recognizer = TextRecognizer(model_file=args.model,
                                device=args.device,
                                compiled_batchsize=compiled_batchsize,
                                image_shape=args.image_shape,
                                label_parser=label_parser,
                                logger=logger)

    dataset = IIIKDataSet(data_path=args.dataset,
                          image_shape=args.image_shape,
                          label_parser=label_parser,
                          logger=logger)
    dataloader = DataLoader(dataset,
                            batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=args.num_workers,
                            drop_last=False)
    correct_num = 0.0
    data_num = 0.0
    for data in tqdm(dataloader):
        results = recognizer.forward(data)
        for bs_idx in range(0, data['x'].shape[0]):
            pred = results[bs_idx][0]
            target = data['label'][bs_idx]
            if ignore_space:
                pred = pred.replace(" ", "")
                target = target.replace(" ", "")
            if pred == target:
                correct_num += 1
            data_num += 1
    acc = correct_num / (data_num + 1e-5)

    runtime_info = OrderedDict(
        [('model', args.model),
         ('data_path', args.dataset),
         ('batch_size', args.batch_size),
         ('device', args.device),
         ('acc', acc),
         ])
    final_report(logger, runtime_info)
