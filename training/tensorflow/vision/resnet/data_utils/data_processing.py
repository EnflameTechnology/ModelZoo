# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# !/usr/bin/python
# coding=utf-8

import tensorflow as tf
import os
import re

_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94
_CHANNEL_MEANS = [_R_MEAN, _G_MEAN, _B_MEAN]
# This value is chosen so that for resnet when the size of input image is 224,
# the size of the smallest side after resize will be 256(int(224 * 1.145) = 256)
_ASP_RESIZE_FACTOR = 1.145
_SHUFFLE_BUFFER = 1500


class DataProcessing(object):
    def __init__(self, is_training, data_dir, dtype, params,
                 target_device='/device:CPU:0'):
        self.is_training = is_training
        self.data_dir = data_dir
        self.dtype = dtype
        self.target_device = target_device
        self.debug_mode = params['debug_mode']
        self.output_size = params['output_size']
        self.num_channels = params['num_channels']
        self.num_class = params['num_class']
        self.num_workers = params['hvd_size']
        self.worker_index = params['rank']

    def _get_sorted_filename_list(self):
        filename_list = [os.path.join(self.data_dir, file) for file in
                         os.listdir(self.data_dir) if re.search('-of-', file)]
        sorted_filename_list = sorted(filename_list)

        return sorted_filename_list, len(sorted_filename_list)

    def _parse_example_proto(self, example_serialized):
        image_feature_map = {
            'image/encoded': tf.FixedLenFeature([], dtype=tf.string, default_value=''),
            'image/class/label': tf.FixedLenFeature([], dtype=tf.int64, default_value=-1),
            'image/class/text': tf.FixedLenFeature([], dtype=tf.string, default_value='')}
        image_sparse_float32 = tf.VarLenFeature(dtype=tf.float32)

        image_feature_map.update(
            {k: image_sparse_float32 for k in ['image/object/bbox/xmin',
                                         'image/object/bbox/ymin',
                                         'image/object/bbox/xmax',
                                         'image/object/bbox/ymax']})
        image_features = tf.parse_single_example(example_serialized, image_feature_map)

        label = tf.cast(image_features['image/class/label'], dtype=tf.int32)

        return image_features['image/encoded'], label

    def _decode_crop_and_flip(self, image_buffer):
        min_object_covered = 0.1
        aspect_ratio_range = [0.75, 1.33]
        area_range = [0.05, 1.0]
        max_attempts = 100

        bbox = tf.constant([0.0, 0.0, 1.0, 1.0],
                           dtype=tf.float32,
                           shape=[1, 1, 4])  # From the entire image
        sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
            tf.image.extract_jpeg_shape(image_buffer),
            bounding_boxes=bbox,
            min_object_covered=min_object_covered,
            aspect_ratio_range=aspect_ratio_range,
            area_range=area_range,
            max_attempts=max_attempts,
            use_image_if_no_bounding_boxes=True)
        bbox_begin, bbox_size, _ = sample_distorted_bounding_box

        # Reassemble the bounding box in the format the crop op requires.
        offset_y, offset_x, _ = tf.unstack(bbox_begin)
        target_height, target_width, _ = tf.unstack(bbox_size)
        crop_window = tf.stack(
            [offset_y, offset_x, target_height, target_width])

        # Use the fused decode and crop op here, which is faster than each in series.
        cropped = tf.image.decode_and_crop_jpeg(
            image_buffer, crop_window, channels=self.num_channels)

        # Flip to add a little more random distortion in.
        if not self.debug_mode:
            cropped = tf.image.random_flip_left_right(cropped)
        return cropped

    def _central_crop(self, image, crop_height, crop_width):
        shape = tf.shape(image)
        height, width = shape[0], shape[1]

        amount_to_be_cropped_h = (height - crop_height)
        crop_top = amount_to_be_cropped_h // 2
        amount_to_be_cropped_w = (width - crop_width)
        crop_left = amount_to_be_cropped_w // 2
        return tf.slice(
            image, [crop_top, crop_left, 0], [crop_height, crop_width, -1])

    def _resize_image(self, image, height, width):
        return tf.image.resize_images(
            image, [height, width], method=tf.image.ResizeMethod.BILINEAR,
            align_corners=False)

    def _smallest_size_at_least(self, height, width, resize_min):
        """Computes new shape with the smallest side equal to `smallest_side`.

        Computes new shape with the smallest side equal to `smallest_side` while
        preserving the original aspect ratio.

        Args:
            height: an int32 scalar tensor indicating the current height.
            width: an int32 scalar tensor indicating the current width.
            resize_min: A python integer or scalar `Tensor` indicating the size of
                the smallest side after resize.

        Returns:
          new_height: an int32 scalar tensor indicating the new height.
          new_width: an int32 scalar tensor indicating the new width.
        """
        resize_min = tf.cast(resize_min, tf.float32)

        # Convert to floats to make subsequent calculations go smoothly.
        height, width = tf.cast(height, tf.float32), tf.cast(width, tf.float32)

        smaller_dim = tf.minimum(height, width)
        scale_ratio = resize_min / smaller_dim

        # Convert back to ints to make heights and widths that TF ops will accept.
        new_height = tf.cast(height * scale_ratio, tf.int32)
        new_width = tf.cast(width * scale_ratio, tf.int32)

        return new_height, new_width

    def _aspect_preserving_resize(self, image, resize_min):
        """Resize images preserving the original aspect ratio.

        Args:
            image: A 3-D image `Tensor`.
            resize_min: A python integer or scalar `Tensor` indicating the size of
                the smallest side after resize.

        Returns:
            resized_image: A 3-D tensor containing the resized image.
        """

        shape = tf.shape(image)
        height, width = shape[0], shape[1]

        new_height, new_width = self._smallest_size_at_least(height, width,
                                                             resize_min)

        return self._resize_image(image, new_height, new_width)

    def _mean_image_subtraction(self, image, means, num_channels):
        """Subtracts the given means from each image channel.

        For example:
            means = [123.68, 116.779, 103.939]
            image = _mean_image_subtraction(image, means)

        Note that the rank of `image` must be known.

        Args:
            image: a tensor of size [height, width, C].
            means: a C-vector of values to subtract from each channel.
            num_channels: number of color channels in the image that will be distorted.

        Returns:
            the centered image.

        Raises:
            ValueError: If the rank of `image` is unknown, if `image` has a rank other
                than three or if the number of channels in `image` doesn't match the
                number of values in `means`.
        """
        if image.get_shape().ndims != 3:
            raise ValueError('Input must be of size [height, width, C>0]')

        if len(means) != num_channels:
            raise ValueError('len(means) must match the number of channels')

        # We have a 1-D tensor of means; convert to 3-D.
        means = tf.expand_dims(tf.expand_dims(means, 0), 0)

        return image - means

    def parse_record(self, raw_record):
        """Parses a record containing a training example of an image.

        The input record is parsed into a label and image, and the image is passed
        through preprocessing steps (cropping, flipping, and so on).

        Args:
            raw_record: scalar Tensor tf.string containing a serialized
                Example protocol buffer.

        Returns:
            Tuple with processed image tensor and one-hot-encoded label tensor.
        """
        image_buffer, label = self._parse_example_proto(raw_record)

        if self.is_training and not self.debug_mode:
            image = self._decode_crop_and_flip(image_buffer)
            image = self._resize_image(image, self.output_size,
                                       self.output_size)
        else:
            image = tf.image.decode_jpeg(image_buffer,
                                         channels=self.num_channels)
            image = self._aspect_preserving_resize(image, int(
                _ASP_RESIZE_FACTOR * self.output_size))
            image = self._central_crop(image, self.output_size,
                                       self.output_size)

        image.set_shape([self.output_size, self.output_size, self.num_channels])

        image = self._mean_image_subtraction(image, _CHANNEL_MEANS,
                                             self.num_channels)

        image = tf.cast(image, self.dtype)
        return image, label

    def input_fn(self, batch_size):
        filenames, data_file_number = self._get_sorted_filename_list()
        dataset = tf.data.Dataset.from_tensor_slices(filenames)
        if self.is_training:
            dataset = dataset.shard(num_shards=self.num_workers,
                                    index=self.worker_index)

        seed = 1234 if self.debug_mode else None
        if self.is_training:
            # Shuffle the input files
            dataset = dataset.shuffle(buffer_size=data_file_number, seed=seed)

        # Convert to individual records
        dataset = dataset.flat_map(tf.data.TFRecordDataset)
        dataset = dataset.prefetch(buffer_size=batch_size)

        if self.is_training:
            dataset = dataset.shuffle(buffer_size=_SHUFFLE_BUFFER, seed=seed)

        dataset = dataset.repeat()
        if not self.debug_mode and self.target_device != '/device:CPU:0':
            dataset = dataset.apply(
                tf.data.experimental.copy_to_device(self.target_device))

        num_parallel_batches = 1 if self.debug_mode else 8
        dataset = dataset.apply(
            tf.contrib.data.map_and_batch(
                lambda value: self.parse_record(value),
                batch_size=batch_size,
                num_parallel_batches=num_parallel_batches))

        dataset = dataset.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)

        return dataset

    def get_synth_input_fn(self, batch_size):
        images = tf.cast(tf.random_uniform(
            [batch_size, self.output_size, self.output_size,
             self.num_channels]), self.dtype)
        labels = tf.cast(tf.random_uniform([batch_size]), tf.int32)

        return tf.data.Dataset.from_tensors((images, labels)).repeat()

class CIFARDataProcessing(DataProcessing):
    def __init__(self, is_training, data_dir, dtype, params,
                 target_device = '/device:CPU:0'):
        super(CIFARDataProcessing, self).__init__(is_training, data_dir, dtype, params, target_device)

    def _decode_crop_and_flip(self, image_buffer):
        image = tf.io.decode_jpeg(image_buffer)
        image = tf.image.resize_image_with_crop_or_pad(image, self.output_size + 8, self.output_size + 8)
        image = tf.random_crop(image, [self.output_size, self.output_size, self.num_channels])
        image = tf.image.random_flip_left_right(image)
        return image

    def parse_record(self, raw_record):
        image, label = self._parse_example_proto(raw_record)
        if self.is_training:
            image = self._decode_crop_and_flip(image)
        else:
            image = tf.io.decode_jpeg(image)
        image = tf.image.per_image_standardization(image)
        image.set_shape([self.output_size, self.output_size, self.num_channels])
        image = tf.cast(image, self.dtype)
        return image, label

