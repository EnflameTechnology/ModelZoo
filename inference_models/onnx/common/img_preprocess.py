#
# Copyright 2022 Enflame. All Rights Reserved.
#
import numbers
import numpy as np
from PIL import Image


def img_crop(img, top, left, height, width):
    return img.crop((left, top, left + width, top + height))


def img_center_crop(img, output_size):
    if isinstance(output_size, numbers.Number):
        output_size = (int(output_size), int(output_size))
    image_width, image_height = img.size
    crop_height, crop_width = output_size
    crop_top = int(round((image_height - crop_height) / 2.))
    crop_left = int(round((image_width - crop_width) / 2.))
    return img_crop(img, crop_top, crop_left, crop_height, crop_width)


def img_crop_fraction(img, frac):
    image_width, image_height = img.size
    crop_height = int(image_height * frac)
    crop_width = int(image_width * frac)
    crop_top = int(round((image_height - crop_height) / 2.))
    crop_left = int(round((image_width - crop_width) / 2.))
    return img_crop(img, crop_top, crop_left, crop_height, crop_width)


def img_resize(img, size, interpolation=Image.BILINEAR):
    if isinstance(size, int):
        w, h = img.size
        if (w <= h and w == size) or (h <= w and h == size):
            return img
        if w < h:
            ow = size
            oh = int(size * h / w)
            return img.resize((ow, oh), interpolation)
        else:
            oh = size
            ow = int(size * w / h)
            return img.resize((ow, oh), interpolation)
    else:
        return img.resize(size, interpolation)

def img_normalize(input_data, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    img_data = input_data.astype('float32')
    mean_vec = np.array(mean)
    stddev_vec = np.array(std)
    norm_img_data = np.zeros(img_data.shape).astype('float32')
    for i in range(img_data.shape[2]):
        norm_img_data[:, :, i] = (img_data[:, :, i] / 255 - mean_vec[i]) / stddev_vec[i]
    norm_img_data = np.transpose(norm_img_data, (2, 0, 1))
    return norm_img_data

