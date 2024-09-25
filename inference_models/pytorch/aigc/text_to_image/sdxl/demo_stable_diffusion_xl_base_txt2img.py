import os
import torch
import random
import argparse
from diffusers import DiffusionPipeline
from diffusers.utils.torch_utils import randn_tensor

import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)
from common.get_scheduler import get_scheduler
from common.utils import LoraUtils, replace_prompt_str, prompt_align
import time

def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(
        description="stable diffusion xl base text2image demo",
        add_help=add_help,
    )
    parser.add_argument(
        "--model_dir", required=True, type=str, help="root path to stable diffusion xl base pretrained models"
    )

    parser.add_argument(
        "--device", default='gcu', type=str, choices=['cpu', 'cuda', 'gcu'], help="Which device do you want to run the program on, CPU, GPU, or GCU?"
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
        "--num_images_per_prompt",
        default=1,
        type=int,
        help="number of images that generated from one prompt",
    )

    parser.add_argument(
        "--prompt",
        nargs="*",
        default=["cute dragon creature"],
        help="Text prompt(s) to guide image generation",
    )


    parser.add_argument(
        "--prompt_2",
        nargs="*",
        default=[""],
        help="Text prompt(s) to guide image generation",
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
        help="The negative prompt(s) to guide the image generation.",
    )

    parser.add_argument(
        "--seed", default=None, type=int, help="seed for generating random data"
    )

    parser.add_argument(
        "--denoising_steps", default=20, type=int, help="how many steps to run unet"
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
        required=True,
        type=int,
        help="image height, for sdxl",
    )

    parser.add_argument(
        "--image_width",
        required=True,
        type=int,
        help="image width, for sdxl",
    )

    parser.add_argument(
        "--output_dir", required=True, type=str, help="target directory to save generated images"
    )

    return parser

def load_model(model_dir, device):
    if device == "cuda":
        dtype=torch.float16
    elif device == "gcu":
        import torch_gcu
        dtype=torch.float16
    elif device == 'cpu':
        dtype=torch.float32

    pipe = DiffusionPipeline.from_pretrained(model_dir, torch_dtype=dtype)
    pipe.unet.to(memory_format=torch.channels_last)
    pipe.vae.to(memory_format=torch.channels_last)
    pipe = pipe.to(device)
    # if device == "cuda":
    #     pipe.enable_xformers_memory_efficient_attention()
    return pipe

def run_stable_diffusion_xl_base_text2image(
    pipe,
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
    output_dir
):
    config_path = os.path.join(model_dir, "scheduler")
    pipe.scheduler = get_scheduler(scheduler, config_path)

    if seed is None:
        seed = 42

    os.makedirs(output_dir, exist_ok=True)
    cross_attention_kwargs = None
    if(len(lora) > 0):
        if(not merge_lora_flag):
            cross_attention_kwargs={"scale": lora_scale}
            print("lora will be used during inference without merging weights")
        else:
            print("weights of lora have been merged into SD models already")
    batch_size = len(prompt_list)
    prompt_2_list, negative_prompt_list, negative_prompt_2_list = prompt_align(prompt_list, [prompt_2_list, negative_prompt_list, negative_prompt_2_list])

    generator = torch.Generator(device='cpu').manual_seed(seed)
    num_channels_latents = 4
    vae_scale_factor = 8
    shape = (batch_size*num_images_per_prompt, num_channels_latents, image_height // vae_scale_factor, image_width // vae_scale_factor)
    latents = randn_tensor(shape, generator=generator, device=torch.device('cpu'), dtype=torch.float16)
    if device == 'cpu':
        latents = latents.to(torch.float32)
    images = pipe(
                    prompt=prompt_list,
                    prompt_2=prompt_2_list,
                    negative_prompt=negative_prompt_list,
                    negative_prompt_2=negative_prompt_2_list,
                    latents=latents,
                    height=image_height,
                    width=image_width,
                    num_images_per_prompt=num_images_per_prompt,
                    num_inference_steps=denoising_steps,
                    guidance_scale=guidance_scale,
                    cross_attention_kwargs=cross_attention_kwargs
                ).images
    for i, prompt in enumerate(prompt_list):
        prompt_str = replace_prompt_str(prompt)
        for j in range(num_images_per_prompt):
            image = images[i*num_images_per_prompt + j]
            prompt_str = prompt_str[:130]
            img_name = str(seed) + '-prompt' + '_' + str(i)  + '-img' + '_' + str(j)  + '-steps' + '_' + str(denoising_steps) + '-cfg' + '_' + str(guidance_scale)+ '-' + prompt_str + ".png"
            t_save_start = time.time()
            image.save(os.path.join(output_dir, img_name))
            t_save_end = time.time()
            print(f'saving current picture costs time: {t_save_end - t_save_start}')

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    print(f'args: {args}')
    pipe = load_model(args.model_dir, args.device)
    merge_lora_flag = args.merge_lora
    if(len(args.lora) > 0):
        lora_util = LoraUtils(pipe)
        adapter_name_list = lora_util.load_lora(lora_list=args.lora)
        lora_util.set_active_adapters(adapter_name_list, args.adapter_weights)
        merge_lora_flag = lora_util.merge_lora_weights(args.merge_lora, args.lora_scale)

    run_stable_diffusion_xl_base_text2image(
        pipe,
        args.device,
        args.model_dir,
        args.image_height,
        args.image_width,
        args.scheduler,
        args.seed,
        args.lora,
        args.adapter_weights,
        args.lora_scale,
        merge_lora_flag,
        args.prompt,
        args.prompt_2,
        args.negative_prompt,
        args.negative_prompt_2,
        args.guidance_scale,
        args.denoising_steps,
        args.num_images_per_prompt,
        args.output_dir
    )
    if(len(args.lora) > 0):
        if(merge_lora_flag):
            lora_util.unmerge_lora_weights()
            print("lora unmerged successfully!")
        lora_util.unload_lora()
