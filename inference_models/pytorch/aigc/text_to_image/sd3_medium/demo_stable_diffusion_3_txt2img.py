import torch
import os
import argparse
from diffusers import StableDiffusion3Pipeline
from diffusers.utils.torch_utils import randn_tensor
import time
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)
from common.utils import replace_prompt_str, prompt_align


def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(
        description="stable-diffusion-3 text2image demo",
        add_help=add_help,
    )
    parser.add_argument(
        "--model_dir", required=True, type=str, help="root path to stable-diffusion-3 models"
    )

    parser.add_argument(
        "--device", default='gcu', type=str, choices=['cpu', 'cuda', 'gcu'], help="Which device do you want to run the program on, CPU, GPU, or GCU?"
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
        "--seed", default=None, type=int, help="seed for generating random data"
    )

    parser.add_argument(
        "--denoising_steps", default=20, type=int, help="how many steps to run unet"
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
        "--output_dir", required=True, type=str, help="target dir to save generated images"
    )

    return parser

def load_model(model_dir, device):
    dtype=torch.float32
    if device == "cuda":
        dtype=torch.float16
    elif device == "gcu":
        import torch_gcu
        dtype=torch.float16
    elif device == 'cpu':
        dtype=torch.float32

    pipe = StableDiffusion3Pipeline.from_pretrained(model_dir, torch_dtype=dtype)
    pipe = pipe.to(device)
    # if device == "cuda":
    #     pipe.enable_xformers_memory_efficient_attention()
    return pipe

def run_stable_diffusion_3_text2image(
                          pipe,
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
                          output_dir
                          ):
    if seed is None:
        seed = 42
    os.makedirs(output_dir, exist_ok=True)

    prompt_2_list, prompt_3_list, negative_prompt_list, negative_prompt_2_list, negative_prompt_3_list = prompt_align(
        prompt_list, [prompt_2_list, prompt_3_list, negative_prompt_list, negative_prompt_2_list, negative_prompt_3_list])
    batch_size = len(prompt_list)
    generator = torch.Generator(device='cpu').manual_seed(seed)
    num_channels_latents = pipe.transformer.config.in_channels
    vae_scale_factor = pipe.vae_scale_factor
    shape = (batch_size*num_images_per_prompt, num_channels_latents, image_height // vae_scale_factor, image_width // vae_scale_factor)
    latents = randn_tensor(shape, generator=generator, device=torch.device('cpu'), dtype=torch.float16)
    if(device == 'cpu'):
        latents = latents.to(torch.float32)
    images = pipe(
                prompt=prompt_list,
                prompt_2=prompt_2_list,
                prompt_3=prompt_3_list,
                negative_prompt=negative_prompt_list,
                negative_prompt_2=negative_prompt_2_list,
                negative_prompt_3=negative_prompt_3_list,
                latents=latents,
                height=image_height,
                width=image_width,
                num_images_per_prompt=num_images_per_prompt,
                num_inference_steps=denoising_steps,
                guidance_scale=guidance_scale
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
    run_stable_diffusion_3_text2image(
                            pipe,
                            args.device,
                            args.model_dir,
                            args.image_height,
                            args.image_width,
                            args.seed,
                            args.prompt,
                            args.prompt_2,
                            args.prompt_3,
                            args.negative_prompt,
                            args.negative_prompt_2,
                            args.negative_prompt_3,
                            args.guidance_scale,
                            args.denoising_steps,
                            args.num_images_per_prompt,
                            args.output_dir
                            )
