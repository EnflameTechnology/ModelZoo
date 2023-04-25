# -*- coding: UTF-8 -*-
import os
import torch
import numpy as np
import argparse
from tqdm import tqdm
import onnxruntime as ort
import nemo.collections.asr as nemo_asr
from nemo.collections.asr.data.audio_to_text import AudioToCharDataset
from nemo.collections.asr.metrics.wer import WER
from common.utils import get_provider
from common.logger import tops_logger, final_report
from collections import OrderedDict
import json

def get_parser():
    # create parser
    parser = argparse.ArgumentParser()
    # add arguments
    parser.add_argument("--device", choices=["cpu","gpu","gcu"], default="cpu", help="backend to use")
    parser.add_argument("--dataset", type=str, default="./data/test_manifest.json", help="dataset to use")
    parser.add_argument("--model", type=str, default="./model/conformer_small-asr-nvidia-op13-fp32-N.onnx", help="onnx to use")
    parser.add_argument("--padding_mode", type=bool, default=False, help="use the padding mode when use dtu")
    # parse arguments
    opt = parser.parse_args()
    return opt

def to_numpy(tensor):

    """
    convert tensor to numpy
    """
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def setup_transcribe_dataloader(cfg, vocabulary):

    """
    load and process test dataset
    """

    config = {
        'manifest_filepath': cfg['manifestpath'],
        'sample_rate': 16000,
        'labels': vocabulary,
        'batch_size': cfg['batch_size'],
        'trim_silence': True,
        'shuffle': False,
    }
    dataset = AudioToCharDataset(
        manifest_filepath=config['manifest_filepath'],
        labels=config['labels'],
        sample_rate=config['sample_rate'],
    )
    return torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=config['batch_size'],
        collate_fn=dataset.collate_fn,
        drop_last=config.get('drop_last', False),
        shuffle=False,
        num_workers=config.get('num_workers', 0),
        pin_memory=config.get('pin_memory', False),
    )

def run_full_test(nemo_model_name, batch_size, print_predictions, op_args):

    """
    do inference and compute wer score
    """

    test_manifest = op_args.dataset
    onnx_file = op_args.model

    config = {"batch_size": batch_size, "manifestpath": test_manifest}
    model = nemo_asr.models.EncDecCTCModel.restore_from(
        restore_path="./model/"+nemo_model_name+".nemo"
    )
    #backend to use
    providers = get_provider(op_args.device) 
    #ort_session = ort.InferenceSession(onnx_file, providers=EP_list, provider_options=options)
    ort_session = ort.InferenceSession(onnx_file, providers=[providers])
    temporary_datalayer = setup_transcribe_dataloader(config, model.decoder.vocabulary)

    wer_nums = []
    wer_denoms = []

    for test_batch in tqdm(temporary_datalayer):

        #preprocess input data
        processed_signal, processed_signal_len = model.preprocessor(
            input_signal=test_batch[0].to(model.device),
            length=test_batch[1].to(model.device),
        )

        if op_args.device == 'gcu' or op_args.padding_mode:
            padding_len = 1000 - processed_signal.shape[2]
            processed_signal = torch.nn.functional.pad(processed_signal,(0,padding_len), "constant", 0)

        ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(processed_signal)}

        if "conformer" in nemo_model_name:
            ort_inputs[ort_session.get_inputs()[1].name] = to_numpy(
                processed_signal_len
            )
        ologits = ort_session.run(None, ort_inputs)
        alogits = np.asarray(ologits)
        logits = torch.from_numpy(alogits[0])
        greedy_predictions = logits.argmax(dim=-1, keepdim=False)

        #compute wer score
        targets = test_batch[2]
        targets_lengths = test_batch[3]
        model._wer.update(greedy_predictions, targets, targets_lengths)
        _, wer_num, wer_denom = model._wer.compute()
        wer_nums.append(to_numpy(wer_num))
        wer_denoms.append(to_numpy(wer_denom))

    return wer_nums, wer_denoms

if __name__ == "__main__":
    option_args = get_parser()
    logger = tops_logger()
    wer_nums, wer_denoms = run_full_test("stt_en_conformer_ctc_small", batch_size=1, print_predictions=True, op_args=option_args)
    # We need to sum all numerators and denominators first. Then divide.
    wer_score = sum(wer_nums) / sum(wer_denoms)
    print("The inference done successful!")
    print("WER= {}".format(wer_score))

    runtime_info = OrderedDict(
        [('model', option_args.model),
         ('dataset', option_args.dataset),
         ('device', option_args.device),
         ('wer', wer_score)
         ])

    final_report(logger, runtime_info)
