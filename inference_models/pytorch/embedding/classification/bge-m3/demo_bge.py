'''
python3.10 demo_bge.py \
    --model_dir=../bge-m3 \
    --device=gcu \

'''
import torch
import argparse
import torch_gcu
from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder

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
        "--dtype",
        type=str,
        default='float16',
        choices=['auto', 'float16', 'bfloat16', 'float', 'float32'],
        help='data type for model weights and activations. '
        'The "auto" option will use FP16 precision '
        'for FP32 and FP16 models, and BF16 precision '
        'for BF16 models.')

    return parser

def test_bi_encoder(
           device,
           model_dir,
           dtype,
           ):
    model = SentenceTransformer(model_dir,device=device)
    model = model.to(dtype)
    # Our sentences we like to encode
    sentences = [
    "This framework generates embeddings for each input sentence",
    "Sentences are passed as a list of strings.",
    "The quick brown fox jumps over the lazy dog.",
    "样例文档-1",
    ]
    # Sentences are encoded by calling model.encode()
    embeddings = model.encode(sentences)
    # Print the embeddings
    for sentence, embedding in zip(sentences, embeddings):
        print("Sentence:", sentence)
        print("Embedding:", embedding)

def test_cross_encoder(
           device,
           model_dir,
           dtype,
           ):
    model = CrossEncoder(model_dir,device=device)
    model.model = model.model.to(dtype)
    query = "Who wrote 'To Kill a Mockingbird'?"
    documents = [
        "'To Kill a Mockingbird' is a novel by Harper Lee published in 1960. It was immediately successful, winning the Pulitzer Prize, and has become a classic of modern American literature.",
        "The novel 'Moby-Dick' was written by Herman Melville and first published in 1851. It is considered a masterpiece of American literature and deals with complex themes of obsession, revenge, and the conflict between good and evil.",
        "Harper Lee, an American novelist widely known for her novel 'To Kill a Mockingbird', was born in 1926 in Monroeville, Alabama. She received the Pulitzer Prize for Fiction in 1961.",
        "Jane Austen was an English novelist known primarily for her six major novels, which interpret, critique and comment upon the British landed gentry at the end of the 18th century.",
        "The 'Harry Potter' series, which consists of seven fantasy novels written by British author J.K. Rowling, is among the most popular and critically acclaimed books of the modern era.",
        "'The Great Gatsby', a novel written by American author F. Scott Fitzgerald, was published in 1925. The story is set in the Jazz Age and follows the life of millionaire Jay Gatsby and his pursuit of Daisy Buchanan."
        ]

    result = model.rank(query, documents, return_documents=True)
    print(result)

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    print(f'args: {args}')
    if args.dtype == "float16":
        dtype=torch.float16
    elif args.dtype == "float32":
        dtype=torch.float32
    else:
        dtype = torch.float16
    if 'bi-encoder' in args.model_type:
        test_bi_encoder(
          device=args.device,
          model_dir=args.model_dir,
          dtype=dtype,
        )
    else:
        test_cross_encoder(
          device=args.device,
          model_dir=args.model_dir,
          dtype=dtype,
        )
