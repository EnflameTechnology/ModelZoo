import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import os
import argparse
import os.path as osp
import torchmetrics
from torchmetrics.text import CharErrorRate
from collections import OrderedDict
from logger import tops_logger, final_report
from datasets import Dataset, Audio
from tqdm import tqdm
from transformers.pipelines.pt_utils import KeyDataset
import time

def read_lists(data_path):
    lists = []
    labels= []
    with open(osp.join(data_path,'dev_clean_test.txt'), 'r', encoding='utf8') as fin:
        for line in fin:
            lists.append(line.split(',')[0])
            labels.append(line.split(',')[-1].strip())
    return lists, labels

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='', help='model path.')
    parser.add_argument('--data_path', default='', help='dataset path.')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for processing.')
    parser.add_argument('--device', type=str, default='gcu', choices=['gcu', 'gpu', 'cpu'], help="Device type for execution.")
    parser.add_argument("--early_stop",action="store_true",help="early stop for perf test")

    return parser.parse_args()

def main():
    args = parse_arguments()
    logger = tops_logger('whisper_large_v3.log')
    if args.device=='gpu':
        device = torch.device('cuda')
        torch_dtype = torch.float16
    elif args.device=='gcu':
        import torch_gcu
        device = torch.device('gcu')
        torch_dtype = torch.float16
    else:
        device = torch.device('cpu')
        torch_dtype = torch.float32

    model = AutoModelForSpeechSeq2Seq.from_pretrained(args.model_path, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True)
    model.to(device)
    processor = AutoProcessor.from_pretrained(args.model_path)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        chunk_length_s=30,
        return_timestamps=True,
        torch_dtype=torch_dtype,
        device=device,
    )

    lists, labels = read_lists(args.data_path)
    wav_files= [osp.join(args.data_path,line).strip() for line in lists]
    dataset = Dataset.from_dict({"audio": wav_files}).cast_column("audio", Audio(sampling_rate=16000))
    pred_list = []
    cer = CharErrorRate()
    logger.info('test start...')
    start = time.time()
    with torch.no_grad():
        i=0
        for results in tqdm(pipe(KeyDataset(dataset, "audio"), batch_size=args.batch_size, generate_kwargs={"language": "english"})):
            if args.early_stop and i >= 5:
                break
            pred_list.append(results['text'].upper())
            i+=1
    end = time.time()
    total_time = end - start
    total_samples = len(pred_list)
    accuracy = 1 - cer(pred_list, labels)
    total_time = end - start
    logger.info('test done...')
    runtime_info = OrderedDict(
            [('model', 'whisper_large_v3'),
            ('batch_size', args.batch_size),
            ('device', args.device),
            ('total_time', total_time),
            ('total_samples', total_samples),
            ('average latency', float(1000 * total_time) / total_samples),
            ('throughput', total_samples / total_time),
            ('process audio time per second', 3600*8 / total_time),
            ('accuracy', accuracy.item()),
            ])
    final_report(logger, runtime_info)

if __name__ == '__main__':
    main()
