'''
export max_len=512
export bs=32
python3.10 precision_test.py \
        --device  gcu \
        --model_dir m3e-base \
        --dataset precisioin_test.json \
        --max_len $max_len \
        --bs $bs
'''
import argparse
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder

from time import perf_counter
import json
import jsonlines

def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(
        description="bge demo",
        add_help=add_help,
    )
    parser.add_argument(
        "--device",
        default="gcu",
        choices=["cpu", "cuda", "gcu"],
        help="Specify the device for testing (cpu, gpu, gcu)")
    parser.add_argument(
        "--model_dir",
        required=True,
        type=str,
        default="./m3e-base",
        help="Specify the model_dir of pretrained_model")
    parser.add_argument(
        "--model_type",
        type=str,
        default="bi-encoder",
        help="Specify the model type, bi-encoder or cross-encoder")
    parser.add_argument(
        "--dataset",
        required=True,
        type=str,
        default="./test.json",
        help="Specify data for test")
    parser.add_argument(
        "--max_len",
        required=True,
        type=int,
        default=128,
        help="Specify max_len of input")
    parser.add_argument(
        "--bs",
        required=True,
        type=int,
        default=1,
        help="Specify batch size")
    parser.add_argument(
            '--dtype',
            type=str,
            default='float16',
            choices=['auto', 'float16', 'bfloat16', 'float', 'float32'],
            help='data type for model weights and activations. '
            'The "auto" option will use FP16 precision '
            'for FP32 and FP16 models, and BF16 precision '
            'for BF16 models.')
    return parser
def precision_bi_encoder(
           device,
           model_dir,
           dataset,
           max_len,
           bs,
           dtype,
           ):
    bs_cnt = 0
    sentences=[]
    with jsonlines.open(dataset) as f:
        for line in f.iter():
            print(line['sentence'])
            sentences.append(line['sentence'])
            bs_cnt = bs_cnt + 1
            if bs_cnt == bs:
                break
#cpu
    model = SentenceTransformer(model_dir,device='cpu')
    model.max_seq_length = max_len
    embeddings_cpu = model.encode(sentences)
#gcu or cuda
    model = SentenceTransformer(model_dir,device=device)
    model = model.to(dtype)
    model.max_seq_length = max_len
    embeddings_gcu = model.encode(sentences)
#CosineSimilarity
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    similarity = cos(torch.from_numpy(embeddings_cpu),torch.from_numpy(embeddings_gcu))
    print('model',model_dir)
    print('max length',max_len)
    print('batch size',bs)
    print('dtype',dtype)
    print('similiarity',similarity)
    print('averge similiarity',torch.mean(similarity))
def precision_cross_encoder(
           device,
           model_dir,
           dataset,
           max_len,
           bs,
           dtype,
           ):
    bs_cnt = 0
    import datasets
    dataset = datasets.load_from_disk(dataset)
    dev = dataset['dev']
    qps=[]
    qns=[]
    for idx in range(bs):
        q=dev[idx]['query']
        p=dev[idx]['positive']
        n=dev[idx]['negative']
        qp=[]
        qn=[]
        qp.append(str(q))
        qp.append(str(p))
        qn.append(str(q))
        qn.append(str(n))
        qps.append(qp)
        qns.append(qn)
    model = CrossEncoder(model_dir,device='cpu')
    model.model = model.model.to(torch.float32)
    model.max_seq_length = max_len
    scores_qps = model.predict(qps,batch_size=bs)
    scores_qns = model.predict(qns,batch_size=bs)
    print('scores cpu',torch.mean(torch.from_numpy(scores_qps)),torch.mean(torch.from_numpy(scores_qns)))
#gcu or cuda
    model = CrossEncoder(model_dir,device=device)
    model.model = model.model.to(dtype)
    model.max_seq_length = max_len
    scores_qps_gcu = model.predict(qps,batch_size=bs)
    scores_qns_gcu = model.predict(qns,batch_size=bs)
    print('scores gcu',torch.mean(torch.from_numpy(scores_qps_gcu)),torch.mean(torch.from_numpy(scores_qns_gcu)))
#Diff
    diffp = scores_qps_gcu - scores_qps
    diffn = scores_qns_gcu - scores_qns
    print('model',model_dir)
    print('max length',max_len)
    print('batch size',bs)
    print('dtype',dtype)
    print('scores_diff',torch.mean(torch.from_numpy(diffp)),torch.mean(torch.from_numpy(diffn)))
if __name__ == "__main__":
    args = get_args_parser().parse_args()
    print(f'args: {args}')
    if args.device == 'gcu':
        import torch_gcu
    if args.dtype == "float16":
        dtype=torch.float16
    elif args.dtype == "float32":
        dtype=torch.float32
    else:
        dtype = torch.float16
    if 'bi-encoder' in args.model_type:
        precision_bi_encoder(device=args.device,
                model_dir=args.model_dir,
                dataset = args.dataset,
                max_len = args.max_len,
                bs = args.bs,
                dtype = dtype,
                )
    else:
        precision_cross_encoder(device=args.device,
              model_dir=args.model_dir,
              dataset = args.dataset,
              max_len = args.max_len,
              bs = args.bs,
              dtype = dtype,
              )
