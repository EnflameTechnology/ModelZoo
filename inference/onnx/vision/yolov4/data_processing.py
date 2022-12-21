#
# SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import math
import cv2
import numpy as np
import os
from common.dataset import Dataset

CATEGORY_NUM = 80

CLASSMAP = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46,
            47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]

class ImageSet(Dataset):
    """A simple class for loading images with PIL and reshaping them to the specified
    input resolution for YOLOv4-608.
    """

    def __init__(self, stride,path,batch_size,yolo_input_resolution, pad=0.5, is_unified_shape = True, augment=False, letter_box = False):
        super().__init__()
        """Initialize with the input resolution for YOLOv4, which will stay fixed in this sample.

        Keyword arguments:
        yolo_input_resolution -- two-dimensional tuple with the target network's (spatial)
        input resolution in HW order
        """
        self.yolo_input_resolution = yolo_input_resolution
        self.rect = True
        self.stride = stride
        self.path = path
        self.batch_size = batch_size
        self.pad = pad
        self.is_unified_shape = is_unified_shape

        self.img_files, self.shapes = self.get_images_path_and_shape(self.path,is_unified_shape)
        self.shapes = np.array(self.shapes, dtype=np.float64)

        n = len(self.img_files)  # number of images
        self.imgs = [None]*n
        batch_index = np.floor(np.arange(n) / self.batch_size).astype(np.int)  # batch index
        nbatch = batch_index[-1] + 1  # number of batches
        self.batch = batch_index  # batch index of image
        self.n = n
        self.indices = range(n)
        self.augment = augment
        self.letter_box = letter_box

        if((not is_unified_shape) and self.rect):
            wh_s = self.image_shapes  # wh
            h_ = wh_s[:, 1]
            w_ = wh_s[:, 0]
            ar_s = h_ / w_  # aspect ratio
            rect_list = ar_s.argsort()
            self.img_files = [self.img_files[i] for i in rect_list]
            self.image_shapes = wh_s[rect_list]  # wh
            ar_s = ar_s[rect_list]

            new_shapes = [[1, 1]] * nbatch
            for i in range(nbatch):
                ar_index = ar_s[batch_index == i]
                min_index, max_index = ar_index.min(), ar_index.max()
                if max_index < 1:
                    new_shapes[i] = [max_index, 1]
                elif min_index > 1:
                    new_shapes[i] = [1, 1 / min_index]

            self.batch_shapes = np.ceil(np.array(new_shapes) * self.img_size / self.stride + self.pad)
            self.batch_shapes = self.batch_shapes.astype(np.int) * self.stride

    def get_images_path_and_shape(self, path, is_unified_shape):
        image_name_list = os.listdir(path)
        shapes = []
        img_paths = []
        for i in range(len(image_name_list)):
            image_path = os.path.join(path,image_name_list[i])
            img_paths.append(image_path)
            if(is_unified_shape):
                continue
            img = cv2.imread(image_path)
            shapes.append((img.shape[1],img.shape[0]))
        return img_paths, shapes

    def load_and_resize(self, index):
        """Load an image from the specified path and resize it to the input resolution.
        Return the input image before resizing as a PIL Image (required for visualization),
        and the resized image as a NumPy float array.

        Keyword arguments:
        input_image_path -- string path of the image to be loaded
        """
        input_image_path = self.img_files[index]
        image_raw = cv2.imread(input_image_path)
        new_resolution = (
            self.yolo_input_resolution[1],
            self.yolo_input_resolution[0])

        if self.letter_box:
            img_h, img_w, _ = image_raw.shape
            new_h, new_w = self.yolo_input_resolution[0], self.yolo_input_resolution[1]
            offset_h, offset_w = 0, 0
            if (new_w / img_w) <= (new_h / img_h):
                new_h = int(img_h * new_w / img_w)
                offset_h = (self.yolo_input_resolution[0] - new_h) // 2
            else:
                new_w = int(img_w * new_h / img_h)
                offset_w = (self.yolo_input_resolution[1] - new_w) // 2
            resized = cv2.resize(image_raw, (new_w, new_h))
            image_resized = np.full((self.yolo_input_resolution[0], self.yolo_input_resolution[1], 3), 127, dtype=np.uint8)
            image_resized[offset_h:(offset_h + new_h), offset_w:(offset_w + new_w), :] = resized
        else:
            image_resized = cv2.resize(image_raw, new_resolution)
        return image_raw, image_resized

    def _shuffle_and_normalize(self, image):
        """Normalize a NumPy array representing an image to the range [0, 1], and
        convert it from HWC format ("channels last") to NCHW format ("channels first"
        with leading batch dimension).

        Keyword arguments:
        image -- image as three-dimensional NumPy float array, in HWC format
        """
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.transpose((2, 0, 1)).astype(np.float32)
        image /= 255.0
        return image

    def __getitem__(self, index):
        # Load image
        raw_image, image_resized= self.load_and_resize(index)
        image_preprocessed = self._shuffle_and_normalize(image_resized)
        raw_shape = raw_image.shape

        image_preprocessed = np.ascontiguousarray(image_preprocessed)

        return image_preprocessed, self.img_files[index], raw_shape

    def __len__(self):
        return len(self.img_files)

    @staticmethod
    def collate_fn(batch):
        img, path, shapes = zip(*batch)  # transposed
        return np.stack(img, 0), path, shapes

class PostprocessYOLO(object):
    """Class for post-processing the three outputs tensors from YOLOv4-608."""

    def __init__(self,
                 yolo_masks,
                 yolo_anchors,
                 obj_threshold,
                 nms_threshold,
                 scale_x_y,
                 letter_box,
                 yolo_input_resolution):
        """Initialize with all values that will be kept when processing several frames.
        Assuming 3 outputs of the network in the case of (large) YOLOv4.

        Keyword arguments:
        yolo_masks -- a list of 3 three-dimensional tuples for the YOLO masks
        yolo_anchors -- a list of 9 two-dimensional tuples for the YOLO anchors
        object_threshold -- threshold for object coverage, float value between 0 and 1
        nms_threshold -- threshold for non-max suppression algorithm,
        float value between 0 and 1
        input_resolution_yolo -- two-dimensional tuple with the target network's (spatial)
        input resolution in HW order
        """
        self.masks = yolo_masks
        self.anchors = yolo_anchors
        self.object_threshold = obj_threshold
        self.nms_threshold = nms_threshold
        self.scale_x_y = scale_x_y
        self.letter_box = letter_box
        self.input_resolution_yolo = yolo_input_resolution

    def process(self, outputs, resolution_raw):
        """Take the YOLOv4 outputs generated from a TensorRT forward pass, post-process them
        and return a list of bounding boxes for detected object together with their category
        and their confidences in separate lists.

        Keyword arguments:
        outputs -- outputs from a TensorRT engine in NCHW format
        resolution_raw -- the original spatial resolution from the input PIL image in WH order
        """
        
        outputs_reshaped = list()
        for output in outputs:
            output = output[np.newaxis,:,:,:]
            outputs_reshaped.append(self._reshape_output_0(output))

        boxes, categories, confidences = self._process_yolo_output(
            outputs_reshaped, resolution_raw)
        if self.letter_box:
            if boxes is not None:
                boxes = self._scale_coords(boxes, self.input_resolution_yolo, resolution_raw)
        return boxes, categories, confidences
    
    def _reshape_output(self, output):
        """Reshape a TensorRT output from NCHW to NHWC format (with expected C=255),
        and then return it in (height,width,3,85) dimensionality after further reshaping.

        Keyword argument:
        output -- an output from a TensorRT engine after inference
        """
        output = np.transpose(output, (1, 2, 0))
        height, width, _ = output.shape
        dim1, dim2 = height, width
        dim3 = 3
        # There are CATEGORY_NUM=80 object categories:
        dim4 = (4 + 1 + CATEGORY_NUM)
        return np.reshape(output, (dim1, dim2, dim3, dim4))

    def _reshape_output_0(self, output):
        """Reshape a TensorRT output from NCHW to NHWC format (with expected C=255),
        and then return it in (height,width,3,85) dimensionality after further reshaping.

        Keyword argument:
        output -- an output from a TensorRT engine after inference
        """
        output = np.transpose(output, [0, 2, 3, 1])
        _, height, width, _ = output.shape
        dim1, dim2 = height, width
        dim3 = 3
        # There are CATEGORY_NUM=80 object categories:
        dim4 = (4 + 1 + CATEGORY_NUM)
        return np.reshape(output, (dim1, dim2, dim3, dim4))

    def xyxy2xywh(self, box1):
        box2 = np.copy(box1)
        x1, y1, x2, y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        box2[:, 0] = x1
        box2[:, 1] = y1
        box2[:, 2] = x2 - x1
        box2[:, 3] = y2 - y1
        return box2

    def xywh2xyxy(self, box1):
        box2 = np.copy(box1)
        x, y, w, h = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        box2[:, 0] = x
        box2[:, 1] = y
        box2[:, 2] = x + w
        box2[:, 3] = y + h
        return box2

    def _scale_coords(self, boxes, yolo_input_resolution, resolution_raw):
        # Rescale coords (xyxy) from img1_shape to img0_shape
        height, width, _ = resolution_raw
        input_height, input_width = yolo_input_resolution[0], yolo_input_resolution[1] 
        boxes = self.xywh2xyxy(boxes)
        gain = min(input_height / height, input_width / width)  # gain  = old / new
        pad = (input_width - width * gain) / 2, (input_height - height * gain) / 2  # wh padding

        boxes[:, [0, 2]] -= pad[0]  # x padding
        boxes[:, [1, 3]] -= pad[1]  # y padding
        boxes[:, :4] /= gain
        # self.clip_coords(boxes, [height, width])
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, width)  # x1, x2
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, height)  # y1, y2
        boxes = self.xyxy2xywh(boxes)
        return boxes

    def _process_yolo_output(self, outputs_reshaped, resolution_raw):
        """Take in a list of three reshaped YOLO outputs in (height,width,3,85) shape and return
        return a list of bounding boxes for detected object together with their category and their
        confidences in separate lists.

        Keyword arguments:
        outputs_reshaped -- list of three reshaped YOLO outputs as NumPy arrays
        with shape (height,width,3,85)
        resolution_raw -- the original spatial resolution from the input PIL image in WH order
        """

        # E.g. in YOLOv4-608, there are three output tensors, which we associate with their
        # respective masks. Then we iterate through all output-mask pairs and generate candidates
        # for bounding boxes, their corresponding category predictions and their confidences:
        boxes, categories, confidences = list(), list(), list()
        for output, mask in zip(outputs_reshaped, self.masks):
            box, category, confidence = self._process_feats(output, mask)
            box, category, confidence = self._filter_boxes(box, category, confidence)
            boxes.append(box)
            categories.append(category)
            confidences.append(confidence)

        boxes = np.concatenate(boxes)
        categories = np.concatenate(categories)
        confidences = np.concatenate(confidences)

        # Scale boxes back to original image shape:
        height, width, _ = resolution_raw
        image_dims = [width, height, width, height]

        if self.letter_box:
            input_h, input_w = self.input_resolution_yolo
            image_dims = [input_w, input_h, input_w, input_h]
        boxes = boxes * image_dims
        # Using the candidates from the previous (loop) step, we apply the non-max suppression
        # algorithm that clusters adjacent bounding boxes to a single bounding box:
        nms_boxes, nms_categories, nscores = list(), list(), list()
        for category in set(categories):
            idxs = np.where(categories == category)
            box = boxes[idxs]
            category = categories[idxs]
            confidence = confidences[idxs]

            # keep = self._nms_boxes(box, confidence)
            keep = self._diounms_sort(box, confidence)

            nms_boxes.append(box[keep])
            nms_categories.append(category[keep])
            nscores.append(confidence[keep])

        if not nms_categories and not nscores:
            return None, None, None

        boxes = np.concatenate(nms_boxes)
        categories = np.concatenate(nms_categories)
        confidences = np.concatenate(nscores)

        return boxes, categories, confidences

    def _process_feats(self, output_reshaped, mask):
        """Take in a reshaped YOLO output in height,width,3,85 format together with its
        corresponding YOLO mask and return the detected bounding boxes, the confidence,
        and the class probability in each cell/pixel.

        Keyword arguments:
        output_reshaped -- reshaped YOLO output as NumPy arrays with shape (height,width,3,85)
        mask -- 2-dimensional tuple with mask specification for this output
        """

        # Two in-line functions required for calculating the bounding box
        # descriptors:
        def sigmoid(value):
            """Return the sigmoid of the input."""
            return 1.0 / (1.0 + math.exp(-value))

        def exponential(value):
            """Return the exponential of the input."""
            return math.exp(value)

        # Vectorized calculation of above two functions:
        sigmoid_v = np.vectorize(sigmoid)
        exponential_v = np.vectorize(exponential)

        grid_h, grid_w, _, _ = output_reshaped.shape

        anchors = [self.anchors[i] for i in mask]

        # Reshape to N, height, width, num_anchors, box_params:
        anchors_tensor = np.reshape(anchors, [1, 1, len(anchors), 2])
        box_xy = sigmoid_v(output_reshaped[..., :2]) * self.scale_x_y - 0.5 * (self.scale_x_y - 1)
        box_wh = exponential_v(output_reshaped[..., 2:4]) * anchors_tensor
        box_confidence = sigmoid_v(output_reshaped[..., 4])

        box_confidence = np.expand_dims(box_confidence, axis=-1)
        box_class_probs = sigmoid_v(output_reshaped[..., 5:])

        col = np.tile(np.arange(0, grid_w), grid_w).reshape(-1, grid_w)
        row = np.tile(np.arange(0, grid_h).reshape(-1, 1), grid_h)

        col = col.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
        row = row.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
        grid = np.concatenate((col, row), axis=-1)

        box_xy += grid
        box_xy /= (grid_w, grid_h)
        box_wh /= self.input_resolution_yolo
        box_xy -= (box_wh / 2.)
        boxes = np.concatenate((box_xy, box_wh), axis=-1)

        # boxes: centroids, box_confidence: confidence level, box_class_probs:
        # class confidence
        return boxes, box_confidence, box_class_probs

    def _filter_boxes(self, boxes, box_confidences, box_class_probs):
        """Take in the unfiltered bounding box descriptors and discard each cell
        whose score is lower than the object threshold set during class initialization.

        Keyword arguments:
        boxes -- bounding box coordinates with shape (height,width,3,4); 4 for
        x,y,height,width coordinates of the boxes
        box_confidences -- bounding box confidences with shape (height,width,3,1); 1 for as
        confidence scalar per element
        box_class_probs -- class probabilities with shape (height,width,3,CATEGORY_NUM)

        """
        box_scores = box_confidences * box_class_probs
        box_classes = np.argmax(box_scores, axis=-1)
        box_class_scores = np.max(box_scores, axis=-1)
        pos = np.where(box_class_scores >= self.object_threshold)

        boxes = boxes[pos]
        classes = box_classes[pos]
        scores = box_class_scores[pos]

        return boxes, classes, scores

    def _nms_boxes(self, boxes, box_confidences):
        """Apply the Non-Maximum Suppression (NMS) algorithm on the bounding boxes with their
        confidence scores and return an array with the indexes of the bounding boxes we want to
        keep (and display later).

        Keyword arguments:
        boxes -- a NumPy array containing N bounding-box coordinates that survived filtering,
        with shape (N,4); 4 for x,y,height,width coordinates of the boxes
        box_confidences -- a Numpy array containing the corresponding confidences with shape N
        """
        x_coord = boxes[:, 0]
        y_coord = boxes[:, 1]
        width = boxes[:, 2]
        height = boxes[:, 3]

        areas = width * height
        ordered = box_confidences.argsort()[::-1]

        keep = list()
        while ordered.size > 0:
            # Index of the current element:
            i = ordered[0]
            keep.append(i)
            xx1 = np.maximum(x_coord[i], x_coord[ordered[1:]])
            yy1 = np.maximum(y_coord[i], y_coord[ordered[1:]])
            xx2 = np.minimum(x_coord[i] + width[i], x_coord[ordered[1:]] + width[ordered[1:]])
            yy2 = np.minimum(y_coord[i] + height[i], y_coord[ordered[1:]] + height[ordered[1:]])

            width1 = np.maximum(0.0, xx2 - xx1 + 1)
            height1 = np.maximum(0.0, yy2 - yy1 + 1)
            intersection = width1 * height1
            union = (areas[i] + areas[ordered[1:]] - intersection)

            # Compute the Intersection over Union (IoU) score:
            iou = intersection / union

            # The goal of the NMS algorithm is to reduce the number of adjacent bounding-box
            # candidates to a minimum. In this step, we keep only those elements whose overlap
            # with the current bounding box is lower than the threshold:
            indexes = np.where(iou <= self.nms_threshold)[0]
            ordered = ordered[indexes + 1]

        keep = np.array(keep)
        return keep

    def _diounms_sort(self, boxes, box_confidences):
        """Apply the Non-Maximum Suppression (NMS) algorithm on the bounding boxes with their
        confidence scores and return an array with the indexes of the bounding boxes we want to
        keep (and display later).

        Keyword arguments:
        boxes -- a NumPy array containing N bounding-box coordinates that survived filtering,
        with shape (N,4); 4 for x,y,height,width coordinates of the boxes
        box_confidences -- a Numpy array containing the corresponding confidences with shape N
        """
        x_coord = boxes[:, 0]
        y_coord = boxes[:, 1]
        width = boxes[:, 2]
        height = boxes[:, 3]

        areas = width * height
        ordered = box_confidences.argsort()[::-1]

        keep = list()
        while ordered.size > 0:
            # Index of the current element:
            i = ordered[0]
            keep.append(i)
            xx1 = np.maximum(x_coord[i], x_coord[ordered[1:]])
            yy1 = np.maximum(y_coord[i], y_coord[ordered[1:]])
            xx2 = np.minimum(x_coord[i] + width[i], x_coord[ordered[1:]] + width[ordered[1:]])
            yy2 = np.minimum(y_coord[i] + height[i], y_coord[ordered[1:]] + height[ordered[1:]])

            width1 = np.maximum(0.0, xx2 - xx1 + 1)
            height1 = np.maximum(0.0, yy2 - yy1 + 1)
            intersection = width1 * height1

            c_xx1 = np.minimum(x_coord[i], x_coord[ordered[1:]])
            c_yy1 = np.minimum(y_coord[i], y_coord[ordered[1:]])
            c_xx2 = np.maximum(x_coord[i] + width[i], x_coord[ordered[1:]] + width[ordered[1:]])
            c_yy2 = np.maximum(y_coord[i] + height[i], y_coord[ordered[1:]] + height[ordered[1:]])

            w = c_xx2 - c_xx1
            h = c_yy2 - c_yy1
            c = w * w + h * h

            union = (areas[i] + areas[ordered[1:]] - intersection)

            # Compute the Intersection over Union (IoU) score:
            iou = intersection / union
            if c.any() == 0:
                indexes = np.where(iou <= self.nms_threshold)[0]
            else:
                ax = x_coord[i] + width[i] / 2
                ay = y_coord[i] + height[i] / 2
                bx = x_coord[ordered[1:]] + width[ordered[1:]] / 2
                by = y_coord[ordered[1:]] + height[ordered[1:]] / 2

                d = (ax - bx) * (ax - bx) + (ay - by) * (ay - by)
                u = pow(d / c, 0.6)
                diou_term = u

                giou = iou - diou_term
                # The goal of the NMS algorithm is to reduce the number of adjacent bounding-box
                # candidates to a minimum. In this step, we keep only those elements whose overlap
                # with the current bounding box is lower than the threshold:
                indexes = np.where(giou <= self.nms_threshold)[0]
            ordered = ordered[indexes + 1]

        keep = np.array(keep)
        return keep
