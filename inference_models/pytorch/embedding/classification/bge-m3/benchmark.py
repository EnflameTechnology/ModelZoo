import argparse
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder
import random
from time import perf_counter
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
        "--max_len",
        required=True,
        type=int,
        default=128,
        help="Specify the max length of input")

    parser.add_argument(
        "--bs",
        required=True,
        type=int,
        default=1,
        help="Specify the batch size")

    parser.add_argument(
         "--warmup_count",
         default=5,
         type=int,
         help="Times of inference execution are needed for preheating/warmup",)

    parser.add_argument(
          "--eval_count",
          default=5,
          type=int,
          help="Times of inference execution are needed for performance evaluation",)

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

def benchmark_bi_encoder(
           device,
           model_dir,
           max_len,
           bs,
           dtype,
           warmup_count,
           eval_count,
           ):
    sentence="This framework generates embeddings for each input sentence"*2000
    sentences=[]
    for i in range(bs):
        sentences.append(sentence)

    model = SentenceTransformer(model_dir,device=device)
    model = model.to(dtype)
    # Our sentences we like to encode
    model.max_seq_length = max_len

    for i in range(warmup_count):
        embeddings = model.encode(sentences)
    if device == "cuda" :
        torch.cuda.synchronize()
    elif device == "gcu":
        torch.gcu.synchronize()

    time_start = perf_counter()

    for i in range(eval_count):
        embeddings = model.encode(sentences)

    if device == "cuda" :
        torch.cuda.synchronize()
    elif device == "gcu":
        torch.gcu.synchronize()
    time_end = perf_counter()
    dif = time_end - time_start
    dif = dif/eval_count

    print('model',model_dir)
    print('max length',max_len)
    print('batch size',bs)
    print('dtype',dtype)
    print('average time per step(s)',dif)

def benchmark_cross_encoder(
           device,
           model_dir,
           max_len,
           bs,
           dtype,
           warmup_count,
           eval_count,
           ):
    
    model = CrossEncoder(model_dir, device=device, max_length = max_len)
    model.model = model.model.to(dtype)
    sentence1="This framework generates embeddings for each input sentence"*512
    sentence2="This is for pair with first sentence This is for pair with "*512
    qps=[]
    for idx in range(bs):
        l = list(sentence1)
        random.shuffle(l)
        q=''.join(l)
        l = list(sentence2)
        random.shuffle(l)
        p=''.join(l)
        qp=[]
        qp.append(str(q))
        qp.append(str(p))
        qps.append(qp)
    for i in range(warmup_count):
        scores = model.predict(qps,batch_size=bs)
    
    dif = 0
    for i in range(eval_count):
        if device == "cuda" :
            torch.cuda.synchronize()
        elif device == "gcu":
            torch.gcu.synchronize()
        time_start = perf_counter()
        scores = model.predict(qps,batch_size=bs)
        if device == "cuda" :
            torch.cuda.synchronize()
        elif device == "gcu":
            torch.gcu.synchronize()
        time_end = perf_counter()
        dif = dif + time_end - time_start
    dif = dif/eval_count

    print('model',model_dir)
    print('max length',max_len)
    print('batch size',bs)
    print('dtype',dtype)
    print('average time per step(s)',dif)
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
        benchmark_bi_encoder(device=args.device,
                model_dir=args.model_dir,
                max_len = args.max_len,
                bs = args.bs,
                dtype = dtype,
                warmup_count = args.warmup_count,
                eval_count = args.eval_count,
                )
    else:
        benchmark_cross_encoder(device=args.device,
                model_dir=args.model_dir,
                max_len = args.max_len,
                bs = args.bs,
                dtype = dtype,
                warmup_count = args.warmup_count,
                eval_count = args.eval_count,
                )
