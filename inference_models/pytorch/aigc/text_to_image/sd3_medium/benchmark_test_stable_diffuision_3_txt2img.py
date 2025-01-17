import os
import json
import time
import torch
import argparse
from diffusers import StableDiffusion3Pipeline

import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)
from common.utils import LoraUtils, prompt_align
from common.get_meta_info import get_meta_info

def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(
        description="stable diffusion 3 txt2img benchmark script",
        add_help=add_help,
    )

    parser.add_argument(
        "--model_dir", required=True, type=str, help="root path to stable diffusion 3 pretrained models"
    )

    parser.add_argument(
        "--device",
        default='gcu',
        type=str,
        choices=['cpu', 'cuda', 'gcu'],
        help="Which device do you want to run the program on, CPU, CUDA, or GCU?"
    )

    parser.add_argument(
        "--num_images_per_prompt",
        default=1,
        type=int,
        help="number of images that generated from one prompt",
    )

    parser.add_argument(
        "--prompt",
        nargs="*",
        default=["a lion sitting on a park bench"],
        help="Text prompt(s) to guide image generation",
    )

    parser.add_argument(
        "--prompt_2",
        nargs="*",
        default=[""],
        help="Text prompt(s) 2 to guide image generation",
    )

    parser.add_argument(
        "--prompt_3",
        nargs="*",
        default=[""],
        help="Text prompt(s) 3 to guide image generation",
    )


    parser.add_argument(
        "--negative_prompt",
        nargs="*",
        default=[""],
        help="The negative prompt(s) to guide the image generation.",
    )

    parser.add_argument(
        "--negative_prompt_2",
        nargs="*",
        default=[""],
        help="The negative prompt(s) 2 to guide the image generation.",
    )

    parser.add_argument(
        "--negative_prompt_3",
        nargs="*",
        default=[""],
        help="The negative prompt(s) 3 to guide the image generation.",
    )

    parser.add_argument(
        "--seed",
        default=42,
        type=int,
        help="seed for generating random data"
    )

    parser.add_argument(
        "--denoising_steps",
        default=20,
        type=int,
        help="how many steps to run unet"
    )

    parser.add_argument(
        "--guidance_scale", default=7.5, type=float, help="guidance_scale or CFG scale"
    )

    parser.add_argument(
        "--image_height",
        required=True,
        type=int,
        help="image height",
    )

    parser.add_argument(
        "--image_width",
        required=True,
        type=int,
        help="image width",
    )

    parser.add_argument(
        "--warmup_count",
        default=3,
        type=int,
        help="times of inference execution are needed for preheating/warmup",
    )

    parser.add_argument(
        "--eval_count",
        default=5,
        type=int,
        help="times of inference execution are needed for performance evaluation",
    )

    parser.add_argument(
        "--output_dir", required=True, type=str, help="target directory to save generated images"
    )

    return parser


def bench_stable_diffusion_3_text2image(
    device,
    model_dir,
    image_height,
    image_width,
    seed,
    prompt_list,
    prompt_2_list,
    prompt_3_list,
    negative_prompt_list,
    negative_prompt_2_list,
    negative_prompt_3_list,
    guidance_scale,
    denoising_steps,
    num_images_per_prompt,
    output_dir,
    warmup_count,
    eval_count,
):
    dtype=torch.float16

    if device == "gcu":
        import torch_gcu
    elif device == 'cpu':
        dtype=torch.float32

    pipe = StableDiffusion3Pipeline.from_pretrained(model_dir, torch_dtype=dtype)
    pipe.transformer.to(memory_format=torch.channels_last)
    pipe.vae.to(memory_format=torch.channels_last)

    pipe = pipe.to(device)

    os.makedirs(output_dir, exist_ok=True)

    generator = torch.Generator(device='cpu').manual_seed(seed)

    prompt_2_list, prompt_3_list, negative_prompt_list, negative_prompt_2_list, negative_prompt_3_list = prompt_align(
        prompt_list, [prompt_2_list, prompt_3_list, negative_prompt_list, negative_prompt_2_list, negative_prompt_3_list])


    infer_pipe = lambda arg1: pipe(
        prompt=prompt_list,
        prompt_2=prompt_2_list,
        prompt_3=prompt_3_list,
        negative_prompt=negative_prompt_list,
        negative_prompt_2=negative_prompt_2_list,
        negative_prompt_3=negative_prompt_3_list,
        num_images_per_prompt=arg1,
        generator=generator,
        num_inference_steps=denoising_steps,
        guidance_scale=guidance_scale,
        height=image_height,
        width=image_width
    ).images

    print("Warming up ...")
    for i in range(warmup_count):
        print(f'warmup: {i+1}/{warmup_count}')
        images = infer_pipe(1)

    total_infer_time = 0
    for iter in range(eval_count):
        t_infer_start = time.time()
        images = infer_pipe(num_images_per_prompt)
        t_infer_end = time.time()

        infer_time = t_infer_end - t_infer_start
        total_infer_time += infer_time

    avg_infer_time = total_infer_time / eval_count

    total_save_time = 0
    for i, image in enumerate(images):
        t_save_start = time.time()
        image.save(os.path.join(output_dir, f"{seed}_{i}.png"))
        t_save_end = time.time()

        save_time = t_save_end - t_save_start
        total_save_time += save_time

    avg_save_time = total_save_time / len(images)

    performance_info_dict = {
        'HxW': f'{image_height}x{image_width}',
        'guidance_scale': guidance_scale,
        'denoising_steps': denoising_steps,
        'num_images_per_prompt': num_images_per_prompt,
        'seed': seed,
        'device': device,
        'pretrained model': model_dir,
        'prompt': prompt_list,
        'prompt_2': prompt_2_list,
        'prompt_3': prompt_3_list,
        'negative_prompt': negative_prompt_list,
        'negative_prompt_2': negative_prompt_2_list,
        'negative_prompt_3': negative_prompt_3_list,
        'output_path': output_dir,
        'warmup_count': warmup_count,
        'eval_count': eval_count,
        "batch size": num_images_per_prompt * len(prompt_list),
        'Avg infer time per batch': avg_infer_time,
        'Avg infer time per image': avg_infer_time / (num_images_per_prompt * len(prompt_list)),
        'Avg save time': avg_save_time,
    }

    meta_info = get_meta_info()
    info_dict = {'performance_info': performance_info_dict}
    info_dict.update(meta_info)

    report_path = os.path.join(output_dir, "report-sd3-txt2img.json")

    with open(report_path, 'w') as f:
        f.write(json.dumps(info_dict, indent=4))

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    print(f'args: {args}')

    bench_stable_diffusion_3_text2image(
        device=args.device,
        model_dir=args.model_dir,
        image_height=args.image_height,
        image_width=args.image_width,
        seed=args.seed,
        prompt_list=args.prompt,
        prompt_2_list=args.prompt_2,
        prompt_3_list=args.prompt_3,
        negative_prompt_list=args.negative_prompt,
        negative_prompt_2_list=args.negative_prompt_2,
        negative_prompt_3_list=args.negative_prompt_3,
        guidance_scale=args.guidance_scale,
        denoising_steps=args.denoising_steps,
        num_images_per_prompt=args.num_images_per_prompt,
        output_dir=args.output_dir,
        warmup_count=args.warmup_count,
        eval_count=args.eval_count
    )
