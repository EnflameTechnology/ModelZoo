import os
import json
import time
import torch
import argparse
from diffusers import StableDiffusionXLPipeline

import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)
from common.get_scheduler import get_scheduler
from common.utils import LoraUtils, prompt_align
from common.get_meta_info import get_meta_info

def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(
        description="stable-diffusion-xl-base-1.0 text2image benchmark",
        add_help=add_help,
    )

    parser.add_argument(
        "--model_dir", required=True, type=str, help="root path to stable-diffusion-xl-base-1.0 pretrained models"
    )

    parser.add_argument(
        "--device",
        default='gcu',
        type=str,
        choices=['cpu', 'cuda', 'gcu'],
        help="Which device do you want to run the program on, CPU, CUDA, or GCU?"
    )

    parser.add_argument(
        "--lora",
        nargs="*",
        default=[],
        type=str,
        help="lora path for merging lora of safetensors format with stable-diffusion model, eg: /path/to/pokemon_lora.safetensors",
    )

    parser.add_argument(
        "--adapter_weights",
        nargs="*",
        default=[],
        type=float,
        help="list of adapter weights",
    )

    parser.add_argument(
        "--lora_scale",
        default=1.0,
        type=float,
        help="coefficient for merging lora with stable-diffusion model, lora_scale must be postitive and usually between 0 and 1",
    )

    parser.add_argument(
        "--merge_lora",
        default=False,
        action='store_true',
        help="whether to merge weights of lora into stable-diffusion model, if true, lora will be merged to Stable-diffusion model with a weight of lora_scale",
    )

    parser.add_argument(
        "--num_images_per_prompt", "-nip",
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
        "--scheduler",
        default="ddim",
        type=str,
        choices=[
            "dpm++_2m",
            "euler_a",
            # "lms",
            # "unipc",
            "ddim",
            # "heun",
            # "dpm2",
            # "dpm2-a",
            "euler",
            "ddpm",
            # "dpm++_2s",
            "pndm",
            # "deis",
            # "dpm++_2m_karras",
            # "dpm++_2m_sde",
            # "lms_karras",
            "dpm_2m_karras",
            # "dpm_sde_karras"
        ],
        help="scheduler name",
    )

    parser.add_argument(
        "--guidance_scale", default=7.5, type=float, help="guidance_scale or CFG scale"
    )

    parser.add_argument(
        "--image_height",
        default=1024,
        type=int,
        help="image height",
    )

    parser.add_argument(
        "--image_width",
        default=1024,
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


def bench_stable_diffusion_xl_base_text2image(
    device,
    model_dir,
    image_height,
    image_width,
    scheduler,
    seed,
    lora,
    adapter_weights,
    lora_scale,
    merge_lora_flag,
    prompt_list,
    prompt_2_list,
    negative_prompt_list,
    negative_prompt_2_list,
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

    pipe = StableDiffusionXLPipeline.from_pretrained(model_dir, torch_dtype=dtype, use_safetensors=True)
    pipe.unet.to(memory_format=torch.channels_last)
    pipe.vae.to(memory_format=torch.channels_last)
    pipe = pipe.to(device)

    cross_attention_kwargs = None
    if(len(lora) > 0):
        lora_util = LoraUtils(pipe)
        adapter_name_list = lora_util.load_lora(lora_list=lora)
        lora_util.set_active_adapters(adapter_name_list, adapter_weights)
        merge_lora_flag = lora_util.merge_lora_weights(merge_lora_flag, lora_scale)
        if(not merge_lora_flag):
            cross_attention_kwargs={"scale": lora_scale}
            print("lora will be used during inference without merging weights")
        else:
            print("weights of lora have been merged into sdxl models already")

    config_path = os.path.join(model_dir, "scheduler")
    pipe.scheduler = get_scheduler(scheduler, config_path)
    os.makedirs(output_dir, exist_ok=True)

    generator = torch.Generator(device='cpu').manual_seed(seed)

    prompt_2_list, negative_prompt_list, negative_prompt_2_list = prompt_align(prompt_list, [prompt_2_list, negative_prompt_list, negative_prompt_2_list])

    infer_pipe = lambda arg1: pipe(
                prompt=prompt_list,
                prompt_2=prompt_2_list,
                negative_prompt=negative_prompt_list,
                negative_prompt_2=negative_prompt_2_list,
                generator=generator,
                height=image_height,
                width=image_width,
                num_images_per_prompt=arg1,
                num_inference_steps=denoising_steps,
                guidance_scale=guidance_scale,
                cross_attention_kwargs=cross_attention_kwargs
                ).images

    print("Warming up ...")
    for i in range(warmup_count):
        print(f'warmup: {i+1}/{warmup_count}')
        images = infer_pipe(1)

    total_infer_time = 0
    print("repeating evaluating ...")
    for iter in range(eval_count):
        print(f'evaluating: {iter+1}/{eval_count}')
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
        'scheduler': scheduler,
        'guidance_scale': guidance_scale,
        'denoising_steps': denoising_steps,
        'num_images_per_prompt': num_images_per_prompt,
        'seed': seed,
        'device': device,
        'pretrained model': model_dir,
        'prompt': prompt_list,
        'prompt_2': prompt_2_list,
        'negative_prompt': negative_prompt_list,
        'negative_prompt_2': negative_prompt_2_list,
        'lora':lora,
        'adapter_weights':adapter_weights,
        'lora_scale':lora_scale,
        'merge_lora':merge_lora_flag,
        'output_path': output_dir,
        'warmup_count': warmup_count,
        'eval_count': eval_count,
        "batch_size": num_images_per_prompt * len(prompt_list),
        'average infer time per batch': avg_infer_time,
        'average infer time per image': avg_infer_time / (num_images_per_prompt * len(prompt_list)),
        'average save time': avg_save_time,
    }

    meta_info = get_meta_info()
    info_dict = {'performance_info': performance_info_dict}
    info_dict.update(meta_info)

    report_path = os.path.join(output_dir, "report-sdxl-base-txt2img.json")

    with open(report_path, 'w') as f:
        f.write(json.dumps(info_dict, indent=4))

    if(len(lora) > 0):
        if(merge_lora_flag):
            lora_util.unmerge_lora_weights()
        lora_util.unload_lora()

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    print(f'args: {args}')

    bench_stable_diffusion_xl_base_text2image(
        device=args.device,
        model_dir=args.model_dir,
        image_height=args.image_height,
        image_width=args.image_width,
        scheduler=args.scheduler,
        seed=args.seed,
        lora=args.lora,
        adapter_weights=args.adapter_weights,
        lora_scale=args.lora_scale,
        merge_lora_flag=args.merge_lora,
        prompt_list=args.prompt,
        prompt_2_list=args.prompt_2,
        negative_prompt_list=args.negative_prompt,
        negative_prompt_2_list=args.negative_prompt_2,
        guidance_scale=args.guidance_scale,
        denoising_steps=args.denoising_steps,
        num_images_per_prompt=args.num_images_per_prompt,
        output_dir=args.output_dir,
        warmup_count=args.warmup_count,
        eval_count=args.eval_count
    )