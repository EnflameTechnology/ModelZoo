#!/usr/bin/env python3
# encoding: utf-8
#
# Copyright 2022 Enflame. All Rights Reserved.
#

import torch
import math
import datasets
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from accelerate import Accelerator
from transformers import GPT2Tokenizer, GPT2LMHeadModel, DefaultDataCollator
from utils.onnx_text_generate import CausalLMModelForOnnxGeneration
from utils.create_batch_data import prepare_datasets
from collections import OrderedDict
from common.logger import tops_logger,final_report

# arguments setting
def build_argparser():
    parser = ArgumentParser()
    args = parser.add_argument_group('Options')
    args.add_argument("--model", type=str,
                      default="./model/gpt2_small-huggingface-op13-fp32-seqN.onnx",
                      help="Required. Path to an .onnx file with a gpt-2 model.")
    args.add_argument("--batchsize", type=int,
                      default=1,
                      help="Required. Batch size to the model with test.")
    args.add_argument("--device", type=str,
                      default="cpu",
                      help="Required. device to the onnx with test.")
    args.add_argument("--dataset", type=str,
                      default="./data",
                      help="Required. Path to the data file with test.")
    return parser

# inference #
def infer(onnx_path, backend, dataset_dir, batch_size):

    '''
    This is onnx inference function for Perplexity testing.
    '''

    model = CausalLMModelForOnnxGeneration.from_pretrained('gpt2', onnx_path, backend)
    data_collator = DefaultDataCollator(return_tensors="np")
    test_datasets = prepare_datasets(dataset_dir)
    test_dataloader = DataLoader(
        test_datasets, collate_fn=data_collator, batch_size=batch_size
    )

    # compute ppl
    logger = tops_logger()
    accelerator = Accelerator()
    losses = []
    for batch in tqdm(test_dataloader):
        if len(batch['input_ids']) == 0:
            continue

        input_ids = batch['input_ids']
        label = batch['labels']
        label = np.expand_dims(label, axis=0)
        outputs = model(input_ids, labels=label)
        neg_log_likelihood = outputs[0]
        losses.append(accelerator.gather(
            neg_log_likelihood.repeat(batch_size)))
    losses = torch.cat(losses)
    losses = losses[: len(test_datasets)]

    try:
        perplexity = math.exp(torch.mean(losses))
    except OverflowError:
        perplexity = float("inf")

    #report json
    runtime_info = OrderedDict(
    [('model', "GPT-2"),
        ('dataset', "wikitext"),
        ('batch_size', batch_size),
        ('device', backend),
        ('perplexity', format(perplexity, '.5f'))
        ])
    final_report(logger, runtime_info)

    return perplexity

def main():
    args = build_argparser().parse_args()
    output_ppl = infer(args.model, args.device, args.dataset, args.batchsize)
    print("=" * 20)
    print(output_ppl)
    print("=" * 20)

if __name__ == '__main__':
    main()