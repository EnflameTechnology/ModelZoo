
#!/usr/bin/python3

"""
Validate a darknet YOLOv4 model accuracy on coco 2017 dataset
"""
import os
import json
import argparse
import numpy as np
import onnxruntime as ort
from os.path import join
from common.utils import get_provider
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from data_processing import PostprocessYOLO, CLASSMAP, ImageSet
from common.logger import tops_logger,final_report
from collections import OrderedDict

from common.dataloader import DataLoader
from tqdm import tqdm
from pathlib import Path


def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(description='Onnx Model Fp32/Fp16 inference',
                                     add_help=add_help)
    parser.add_argument('--model',
                        default='./model/yolov4-leaky-608-darknet-op13-fp32-N.onnx',
                        help='onnx path')
    parser.add_argument('--dataset',
                        default='./data',
                        type=str,
                        help='dataset path')
    parser.add_argument('--device',
                        default='cpu',
                        help='gcu, gpu, cpu')
    parser.add_argument('--batch_size',
                        default=32,
                        type=int,
                        help='batch size')

    parser.add_argument('--input_height',
                        default=608,
                        type=int,
                        help='model input image height')
    parser.add_argument('--input_width',
                        default=608,
                        type=int,
                        help='model input image width')
    parser.add_argument('--conf-thres',
                        type=float,
                        default=0.001,
                        help='confidence threshold')
    parser.add_argument('--iou-thres',
                        type=float,
                        default=0.6,
                        help='NMS IoU threshold')
    parser.add_argument('--scale_x_y',
                        type=float,
                        default=1.05,
                        help='scale_x_y')
    parser.add_argument('--letter-box',
                        dest="letter_box",
                        help="letter_box",
                        action="store_true")
    parser.add_argument('--num_workers',
                        default=8,
                        type=int,
                        help='data loader workers number')
    return parser

def main(args):
    logger = tops_logger()
    count = 1
    jdict = []
    provider = get_provider(args.device)
    session = ort.InferenceSession(args.model, providers=[provider])
    input_name = session.get_inputs()[0].name
    anno = COCO(join(args.dataset, "annotations/instances_val2017.json"))
    input_resolution_HW = (args.input_height, args.input_width)

    stride = 32
    img_path = os.path.join(args.dataset,'val2017')
    dataset = ImageSet(stride=stride,path=img_path,batch_size=args.batch_size,yolo_input_resolution=input_resolution_HW,pad=0.5,is_unified_shape = True, letter_box = args.letter_box)
    dataloader = DataLoader(dataset, batch_size=args.batch_size,shuffle=False, num_workers=args.num_workers, drop_last=False, collate_fn=ImageSet.collate_fn)
    postprocessor_args = {"yolo_masks": [(0, 1, 2), (3, 4, 5), (6, 7, 8)],                    # A list of 3 three-dimensional tuples for the YOLO masks
                            "yolo_anchors": [(12, 16), (19, 36), (40, 28), (36, 75), (76, 55),  # A list of 9 two-dimensional tuples for the YOLO anchors
                                            (72, 146), (142, 110), (192, 243), (459, 401)],
                            "obj_threshold": args.conf_thres,                                               # Threshold for object coverage, float value between 0 and 1
                            "nms_threshold": args.iou_thres,                                               # Threshold for non-max suppression algorithm, float value between 0 and 1
                            "scale_x_y": args.scale_x_y,
                            "letter_box": args.letter_box,
                            "yolo_input_resolution": input_resolution_HW}
    postprocessor = PostprocessYOLO(**postprocessor_args)
    # Do inference
    s = 'scanning images and infering...'
    for batch_i, (imgs, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):
        raw_result = session.run([], {input_name: imgs})
        
        for si, pred in enumerate(zip(raw_result[0], raw_result[1], raw_result[2])):
            boxes, classes, scores = postprocessor.process(pred, (shapes[si]))
            path = Path(paths[si])
            imgId = int(path.stem) if path.stem.isnumeric() else path.stem
            if boxes is not None:
                for box, category, score in zip(boxes, classes, scores):
                    jdict.append({'image_id': imgId,
                    'category_id': CLASSMAP[int(category)],
                    'bbox': box.tolist(),
                    'score': np.round(score, 5).tolist()})

    cocoGt = anno
    cocoDt=cocoGt.loadRes(jdict)
    imgIds=sorted(cocoGt.getImgIds())
    cocoEval = COCOeval(cocoGt,cocoDt,'bbox')
    cocoEval.params.imgIds  = imgIds
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    runtime_info = OrderedDict([
     ('model', args.model),
    ('dataset', args.dataset),
    ('batch_size', args.batch_size),
    ('device', args.device),
    ('conf thres', args.conf_thres),
    ('mAP', cocoEval.stats[0])
    ])

    final_report(logger, runtime_info)

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
