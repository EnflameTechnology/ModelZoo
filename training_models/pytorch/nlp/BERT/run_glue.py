# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
"""BERT finetuning runner."""

from __future__ import absolute_import, division, print_function

import pickle
import argparse
import logging
import os
import random
import collections

import json
import time
import datetime
import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from tqdm import tqdm, trange

from file_utils import PYTORCH_PRETRAINED_BERT_CACHE
import modeling
from tokenization import BertTokenizer
from optimization import BertAdam, warmup_linear
from schedulers import LinearWarmUpScheduler
from tops_models.logger import tops_logger
import tops_models.common_utils as common_utils
from tops_models.time_watcher import EnflameThread
from collections import OrderedDict
from torch.utils.tensorboard import SummaryWriter
try:
    import apex
    from apex import amp
    APEX_IS_AVAILABLE = True
except ImportError:
    print("apex not found")
    APEX_IS_AVAILABLE = False

from sklearn.metrics import matthews_corrcoef, f1_score
from utils import (is_main_process, mkdir_by_main_process, format_step,
                   get_world_size)
from processors.glue import PROCESSORS, convert_examples_to_features

torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if task_name == "cola":
        return {"mcc": matthews_corrcoef(labels, preds)}
    elif task_name == "sst-2":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "mrpc":
        return acc_and_f1(preds, labels)
    elif task_name == "sts-b":
        return pearson_and_spearman(preds, labels)
    elif task_name == "qqp":
        return acc_and_f1(preds, labels)
    elif task_name == "mnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "mnli-mm":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "qnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "rte":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "wnli":
        return {"acc": simple_accuracy(preds, labels)}
    else:
        raise KeyError(task_name)


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }


def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)

def parse_args(parser=argparse.ArgumentParser()):
    ## Required parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the .tsv files (or other data "
        "files) for the task.",
    )
    parser.add_argument(
        "--bert_model",
        default=None,
        type=str,
        required=True,
        help="Bert pre-trained model selected in the list: bert-base-uncased, "
        "bert-large-uncased, bert-base-cased, bert-large-cased, "
        "bert-base-multilingual-uncased, bert-base-multilingual-cased, "
        "bert-base-chinese.",
    )
    parser.add_argument(
        "--task_name",
        default=None,
        type=str,
        required=True,
        choices=PROCESSORS.keys(),
        help="The name of the task to train.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints "
        "will be written.",
    )
    parser.add_argument(
        "--init_checkpoint",
        default=None,
        type=str,
        required=True,
        help="The checkpoint file from pretraining",
    )

    ## Other parameters
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after WordPiece "
        "tokenization. \n"
        "Sequences longer than this will be truncated, and sequences shorter \n"
        "than this will be padded.",
    )
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to get model-task performance on the dev"
                        " set by running eval.")
    parser.add_argument("--do_predict",
                        action='store_true',
                        help="Whether to output prediction results on the dev "
                        "set by running eval.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Batch size per GPU or DTU for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Batch size per GPU or DTU for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps",
                        default=-1.0,
                        type=float,
                        help="Total number of training steps to perform.")
    parser.add_argument(
        "--warmup_proportion",
        default=0.1,
        type=float,
        help="Proportion of training to perform linear learning rate warmup "
        "for. E.g., 0.1 = 10%% of training.",
    )
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus or gcus")
    parser.add_argument('--workers', type=int, default=8, help='maximum number of dataloader workers')
    parser.add_argument('--seed',
                        type=int,
                        default=1,
                        help="random seed for initialization")
    parser.add_argument(
        '--gradient_accumulation_steps',
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a "
        "backward/update pass.")
    parser.add_argument(
        '--fp16',
        action='store_true',
        help="Mixed precision training",
    )
    parser.add_argument(
        '--amp',
        action='store_true',
        help="Mixed precision training",
    )
    parser.add_argument(
        '--loss_scale',
        type=float,
        default=0,
        help="Loss scaling to improve fp16 numeric stability. Only used when "
        "fp16 set to True.\n"
        "0 (default value): dynamic loss scaling.\n"
        "Positive power of 2: static loss scaling value.\n",
    )
    parser.add_argument('--server_ip',
                        type=str,
                        default='',
                        help="Can be used for distant debugging.")
    parser.add_argument('--server_port',
                        type=str,
                        default='',
                        help="Can be used for distant debugging.")
    parser.add_argument('--vocab_file',
                        type=str,
                        default=None,
                        required=True,
                        help="Vocabulary mapping/file BERT was pretrainined on")
    parser.add_argument("--config_file",
                        default=None,
                        type=str,
                        required=True,
                        help="The BERT model config")
    parser.add_argument('--skip_checkpoint',
                        default=False,
                        action='store_true',
                        help="Whether to save checkpoints")
    parser.add_argument('--log_freq',
                        type=int, default=50,
                        help='frequency of logging loss.')
    parser.add_argument("--device", default='gpu', type=str,
                        help="Device to run ,gpu or gcu.")
    parser.add_argument("--use_apex",
                        default=False,
                        action='store_true',
                        help="Whether to use apex")
    parser.add_argument('--skip_steps', default=10, type=int,help='steps to skip while computing fps')
    parser.add_argument('--perf',
                        default=False,
                        action='store_true',
                        help='True to only get total mean fps,False to get every step fps(will have impact on performance')
    return parser.parse_args()


def init_optimizer_and_amp(model, learning_rate, loss_scale, warmup_proportion,
                           num_train_optimization_steps, use_fp16):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [
                p for n, p in param_optimizer
                if not any(nd in n for nd in no_decay)
            ],
            'weight_decay': 0.01
        },
        {
            'params': [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            'weight_decay': 0.0
        },
    ]
    optimizer, scheduler = None, None
    if use_fp16 and APEX_IS_AVAILABLE and args.use_apex:
        logger.info("using fp16")
        try:
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError("Please install apex from "
                              "https://www.github.com/nvidia/apex to use "
                              "distributed and fp16 training.")

        if num_train_optimization_steps is not None:
            optimizer = FusedAdam(
                optimizer_grouped_parameters,
                lr=learning_rate,
                bias_correction=False,
            )
        amp_inits = amp.initialize(
            model,
            optimizers=optimizer,
            opt_level="O2",
            keep_batchnorm_fp32=False,
            loss_scale="dynamic" if loss_scale == 0 else loss_scale,
        )
        model, optimizer = (amp_inits
                            if num_train_optimization_steps is not None else
                            (amp_inits, None))
        if num_train_optimization_steps is not None:
            scheduler = LinearWarmUpScheduler(
                optimizer,
                warmup=warmup_proportion,
                total_steps=num_train_optimization_steps,
            )
    else:
        logger.info("using fp32")
        if num_train_optimization_steps is not None:
            optimizer = BertAdam(
                optimizer_grouped_parameters,
                lr=learning_rate,
                warmup=warmup_proportion,
                t_total=num_train_optimization_steps,
            )
    return model, optimizer, scheduler


def gen_tensor_dataset(features):
    all_input_ids = torch.tensor(
        [f.input_ids for f in features],
        dtype=torch.long,
    )
    all_input_mask = torch.tensor(
        [f.input_mask for f in features],
        dtype=torch.long,
    )
    all_segment_ids = torch.tensor(
        [f.segment_ids for f in features],
        dtype=torch.long,
    )
    all_label_ids = torch.tensor(
        [f.label_id for f in features],
        dtype=torch.long,
    )
    return TensorDataset(
        all_input_ids,
        all_input_mask,
        all_segment_ids,
        all_label_ids,
    )


def get_train_features(data_dir, bert_model, max_seq_length, do_lower_case,
                       local_rank, train_batch_size,
                       gradient_accumulation_steps, num_train_epochs, tokenizer,
                       processor, dist):
    cached_train_features_file = os.path.join(
        data_dir,
        '{0}_{1}_{2}'.format(
            list(filter(None, bert_model.split('/'))).pop(),
            str(max_seq_length),
            str(do_lower_case),
        ),
    )
    train_features = None
    try:
        with open(cached_train_features_file, "rb") as reader:
            train_features = pickle.load(reader)
        logger.info("Loaded pre-processed features from {}".format(
            cached_train_features_file))
    except:
        logger.info("Did not find pre-processed features from {}".format(
            cached_train_features_file))
        train_examples = processor.get_train_examples(data_dir)
        train_features, _ = convert_examples_to_features(
            train_examples,
            processor.get_labels(),
            max_seq_length,
            tokenizer,
        )
        if is_main_process(dist):
            logger.info("  Saving train features into cached file %s",
                        cached_train_features_file)
            with open(cached_train_features_file, "wb") as writer:
                pickle.dump(train_features, writer)
    return train_features


def dump_predictions(path, label_map, preds, examples):
    label_rmap = {label_idx: label for label, label_idx in label_map.items()}
    predictions = {
        example.guid: label_rmap[preds[i]] for i, example in enumerate(examples)
    }
    with open(path, "w") as writer:
        json.dump(predictions, writer)


def main(args):
    tb_writer = SummaryWriter('runs/tensorboard/dtu_{}'.format(args.local_rank if args.local_rank >=0 else 0))
    info_dict = common_utils.tops_models_collection(args.device)
    global_watcher = EnflameThread().watcher(batch_size_per_step=args.train_batch_size,
                                             logger=logger,
                                             skip_steps=args.skip_steps)
    args.fp16 = args.fp16 or args.amp
    if args.server_ip and args.server_port:
        # Distant debugging - see
        # https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        logger.info("Waiting for debugger attach")
        ptvsd.enable_attach(
            address=(args.server_ip, args.server_port),
            redirect_output=True,
        )
        ptvsd.wait_for_attach()
    if args.device.lower() == "gpu":
        import torch.distributed as dist
        from torch.utils.data.distributed import DistributedSampler
        if args.local_rank == -1 or args.no_cuda:
            device = torch.device(
                "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
            n_device = torch.cuda.device_count()
        else:
            torch.cuda.set_device(args.local_rank)
            device = torch.device("cuda", args.local_rank)
            n_device = 1
            # Initializes the distributed backend which will take care of
            # sychronizing nodes/GPUs.
            if not dist.is_initialized():
                dist.init_process_group(backend='nccl')
    else:
        import torch_gcu

        import torch_gcu.distributed as dist
        from torch_gcu.utils.data.distributed import DistributedSampler
        device = torch_gcu.gcu_device(args.local_rank if args.local_rank >=0 else 0)
        if args.distributed and not dist.is_initialized():
            dist.init_process_group(backend='eccl', init_method='env://')
        n_device = 1
    logger.info("device: {} n_device: {}, distributed training: {}, "
                "16-bits training: {}".format(
                    device,
                    n_device,
                    args.distributed,
                    args.fp16,
                ))

    if not args.do_train and not args.do_eval and not args.do_predict:
        raise ValueError("At least one of `do_train`, `do_eval` or "
                         "`do_predict` must be True.")
    if args.max_steps > 0  and args.skip_steps > args.max_steps:
        raise ValueError("max_steps should be large than skip_steps")
    if is_main_process(dist):
        if (os.path.exists(args.output_dir) and os.listdir(args.output_dir) and
                args.do_train):
            logger.warning("Output directory ({}) already exists and is not "
                           "empty.".format(args.output_dir))
    mkdir_by_main_process(args.output_dir, dist)

    logger.info("Config: {}".format([str(args)]))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, "
                         "should be >= 1".format(
                             args.gradient_accumulation_steps))
    if args.gradient_accumulation_steps > args.train_batch_size:
        raise ValueError("gradient_accumulation_steps ({}) cannot be larger "
                         "train_batch_size ({}) - there cannot be a fraction "
                         "of one sample.".format(
                             args.gradient_accumulation_steps,
                             args.train_batch_size,
                         ))
    args.train_batch_size = (args.train_batch_size //
                             args.gradient_accumulation_steps)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device.lower() == "gpu" and n_device > 0:
        torch.cuda.manual_seed_all(args.seed)
    logger.info("SEED: {}".format(args.seed))

    processor = PROCESSORS[args.task_name]()
    num_labels = len(processor.get_labels())

    #tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    tokenizer = BertTokenizer(
        args.vocab_file,
        do_lower_case=args.do_lower_case,
        max_len=512,
    )  # for bert large

    num_train_optimization_steps = None
    if args.do_train:
        train_features = get_train_features(
            args.data_dir,
            args.bert_model,
            args.max_seq_length,
            args.do_lower_case,
            args.local_rank,
            args.train_batch_size,
            args.gradient_accumulation_steps,
            args.num_train_epochs,
            tokenizer,
            processor,
            dist
        )
        num_train_optimization_steps = int(
            len(train_features) / args.train_batch_size /
            args.gradient_accumulation_steps) * args.num_train_epochs
        if args.distributed:
            num_train_optimization_steps = (num_train_optimization_steps //
                                            dist.get_world_size())

    # Prepare model
    config = modeling.BertConfig.from_json_file(args.config_file)
    # Padding for divisibility by 8
    if config.vocab_size % 8 != 0:
        config.vocab_size += 8 - (config.vocab_size % 8)

    modeling.ACT2FN["bias_gelu"] = modeling.bias_gelu_training
    model = modeling.BertForSequenceClassification(
        config,
        num_labels=num_labels,
    )
    logger.info("USING CHECKPOINT from {}".format(args.init_checkpoint))
    model.load_state_dict(
        torch.load(args.init_checkpoint, map_location='cpu')["model"],
        strict=False,
    )
    logger.info("USED CHECKPOINT from {}".format(args.init_checkpoint))
    logger.info("num_parameters: {}".format(
                sum([p.numel() for p in model.parameters() if p.requires_grad])))

    model.to(device)
    # Prepare optimizer
    model, optimizer, scheduler = init_optimizer_and_amp(
        model,
        args.learning_rate,
        args.loss_scale,
        args.warmup_proportion,
        num_train_optimization_steps,
        args.fp16,
    )

    if args.distributed and APEX_IS_AVAILABLE and args.use_apex:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError("Please install apex from "
                              "https://www.github.com/nvidia/apex to use "
                              "distributed and fp16 training.")
        model = DDP(model, find_unused_parameters = True)
    elif args.device == 'gcu' and args.distributed:
        from torch_gcu.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=[args.local_rank], find_unused_parameters = True)
    elif n_device > 1:
        model = torch.nn.DataParallel(model)

    loss_fct = torch.nn.CrossEntropyLoss()

    results = {}
    if args.do_train:
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_features))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)
        train_data = gen_tensor_dataset(train_features)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data, drop_last=True)
        train_dataloader = DataLoader(
            train_data,
            sampler=train_sampler,
            batch_size=args.train_batch_size,
            num_workers = args.workers
        )

        global_step = 0
        tr_loss = 0
        latency_train = 0.0
        nb_tr_examples = 0
        model.train()
        tic_train = time.perf_counter()
        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            for step, batch in enumerate(
                    tqdm(train_dataloader, desc="Iteration")):
                if args.skip_steps > 0 and global_step == args.skip_steps:
                    tic_train = time.perf_counter()
                if args.max_steps > 0 and global_step > args.max_steps:
                    break
                optimizer.zero_grad(True)
                if step % args.log_freq == 0 and args.perf == False:
                    global_watcher.watch("session_start")
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                logits = model(input_ids, segment_ids, input_mask)
                loss = loss_fct(
                    logits.view(-1, num_labels),
                    label_ids.view(-1),
                )
                if n_device > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        # modify learning rate with special warm up for BERT
                        # which FusedAdam doesn't do
                        scheduler.step()

                    if args.device.lower() == "gpu":
                        optimizer.step()
                    else:
                        torch_gcu.optimizer_step(optimizer, [loss, logits],
                                                 mode=torch_gcu.JitRunMode.ASYNC if args.distributed else torch_gcu.JitRunMode.SAFEASYNC,
                                                 model=model)
                    global_step += 1
                if step % args.log_freq == 0 and args.perf == False:
                    global_watcher.watch("session_end")
                    logger.info("step_loss: {},learning_rate: {}".format(loss,optimizer.param_groups[0]['lr']))
                    tb_writer.add_scalar('loss', loss, global_step)
                    tb_writer.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step)
        latency_train = time.perf_counter() - tic_train
        total_time_str = str(datetime.timedelta(seconds=int(latency_train)))
        if args.perf == False:
            train_info = OrderedDict(
            [('model', 'bert'),
            ('train_batch_size', args.train_batch_size),
            ('epochs', args.num_train_epochs),
            ('training_step_per_epoch', len(train_features)),
            ('device', args.device),
            ('skip_steps', args.skip_steps),
            ('train_fps_mean', global_watcher.calc_fps()),
            ('train_fps_min', global_watcher.min_fps()),
            ('train_fps_max', global_watcher.max_fps()),
            ('training_time', total_time_str)
            ])
            info_dict.update(train_info)
        tr_loss = tr_loss / global_step
        if args.max_steps == -1:
            results.update({
                'global_step':
                    global_step,
                'train:loss':
                    tr_loss,
                'train:latency':
                    latency_train,
                'train:num_samples_per_device':
                    nb_tr_examples,
                'train:throughput':
                    (get_world_size() * nb_tr_examples - args.skip_steps*args.train_batch_size) / latency_train,
            })
        else:
            results.update({
                'global_step':
                    global_step,
                'train:loss':
                    tr_loss,
                'train:latency':
                    latency_train,
                'train:num_samples_per_device':
                    global_step*args.train_batch_size,
                'train:throughput':
                    (get_world_size() * global_step*args.train_batch_size - args.skip_steps*args.train_batch_size) / latency_train,
            })
        if is_main_process(dist) and not args.skip_checkpoint:
            model_to_save = model.module if hasattr(model, 'module') else model
            torch.save(
                {"model": model_to_save.state_dict()},
                os.path.join(args.output_dir, modeling.WEIGHTS_NAME),
            )
            with open(
                    os.path.join(args.output_dir, modeling.CONFIG_NAME),
                    'w',
            ) as f:
                f.write(model_to_save.config.to_json_string())

    if (args.do_eval or args.do_predict) and is_main_process(dist):
        eval_examples = processor.get_dev_examples(args.data_dir)
        eval_features, label_map = convert_examples_to_features(
            eval_examples,
            processor.get_labels(),
            args.max_seq_length,
            tokenizer,
        )
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        predict_info = collections.OrderedDict([('predict_num_examples', len(eval_examples)),
                                            ('predict_batch_size', args.eval_batch_size)])
        info_dict.update(predict_info)
        eval_data = gen_tensor_dataset(eval_features)
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(
            eval_data,
            sampler=eval_sampler,
            batch_size=args.eval_batch_size,
        )

        model.eval()
        preds = None
        out_label_ids = None
        eval_loss = 0
        nb_eval_steps, nb_eval_examples = 0, 0
        if args.device.lower() == "gpu":
            cuda_events = [(torch.cuda.Event(enable_timing=True),
                            torch.cuda.Event(enable_timing=True))
                          for _ in range(len(eval_dataloader))]
        for i, (input_ids, input_mask, segment_ids, label_ids) in tqdm(
                enumerate(eval_dataloader),
                desc="Evaluating",
        ):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            label_ids = label_ids.to(device)

            with torch.no_grad():
                if args.device.lower() == "gpu":
                    cuda_events[i][0].record()
                logits = model(input_ids, segment_ids, input_mask)
                if args.device.lower() == "gpu":
                    cuda_events[i][1].record()
                if args.do_eval:
                    eval_loss += loss_fct(
                        logits.view(-1, num_labels),
                        label_ids.view(-1),
                    ).mean().item()

            nb_eval_steps += 1
            nb_eval_examples += input_ids.size(0)
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = label_ids.detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(
                    out_label_ids,
                    label_ids.detach().cpu().numpy(),
                    axis=0,
                )
        if args.device.lower() == "gpu":
            torch.cuda.synchronize()
            eval_latencies = [
                event_start.elapsed_time(event_end)
                for event_start, event_end in cuda_events
            ]
            eval_latencies = list(sorted(eval_latencies))

            def infer_latency_sli(threshold):
                index = int(len(eval_latencies) * threshold) - 1
                index = min(max(index, 0), len(eval_latencies) - 1)
                return eval_latencies[index]

            eval_throughput = (args.eval_batch_size /
                              (np.mean(eval_latencies) / 1000))

            results.update({
                'eval:num_samples_per_gpu': nb_eval_examples,
                'eval:num_steps': nb_eval_steps,
                'infer:latency(ms):50%': infer_latency_sli(0.5),
                'infer:latency(ms):90%': infer_latency_sli(0.9),
                'infer:latency(ms):95%': infer_latency_sli(0.95),
                'infer:latency(ms):99%': infer_latency_sli(0.99),
                'infer:latency(ms):100%': infer_latency_sli(1.0),
                'infer:latency(ms):avg': np.mean(eval_latencies),
                'infer:latency(ms):std': np.std(eval_latencies),
                'infer:latency(ms):sum': np.sum(eval_latencies),
                'infer:throughput(samples/s):avg': eval_throughput,
            })
        preds = np.argmax(preds, axis=1)
        if args.do_predict:
            dump_predictions(
                os.path.join(args.output_dir, 'predictions.json'),
                label_map,
                preds,
                eval_examples,
            )
        if args.do_eval:
            results['eval:loss'] = eval_loss / nb_eval_steps
            eval_result = compute_metrics(args.task_name, preds, out_label_ids)
            results.update(eval_result)

    if is_main_process(dist):
        logger.info("***** Results *****")
        for key in sorted(results.keys()):
            logger.info("  %s = %s", key, str(results[key]))
        with open(os.path.join(args.output_dir, "results.txt"), "w") as writer:
            json.dump(results, writer)
        dllogger_queries_from_results = {
            'exact_match': 'acc',
            'F1': 'f1',
            'e2e_train_time': 'train:latency',
            'training_sequences_per_second': 'train:throughput',
            'e2e_inference_time': ('infer:latency(ms):sum', lambda x: x / 1000),
            'inference_sequences_per_second': 'infer:throughput(samples/s):avg',
        }
        for key, query in dllogger_queries_from_results.items():
            results_key, convert = (query if isinstance(query, tuple) else
                                    (query, lambda x: x))
            if results_key not in results:
                continue
            logger.info("{}:{}".format(key,convert(results[results_key])))
        json_report = json.dumps(info_dict, indent=4, sort_keys=False)
        logger.info("Final report:\n{}".format(json_report))
    return results


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    args = parse_args()
    args.local_rank = int(os.environ['LOCAL_RANK']) if 'LOCAL_RANK' in os.environ else -1
    args.distributed = args.local_rank != -1
    main(args)
