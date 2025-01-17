import torch
import os
import argparse
from diffusers import HunyuanDiTPipeline
import time
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)
from common.utils import replace_prompt_str, prompt_align


def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(
        description="hunyuanDit-v1.2-diffusers text2image demo",
        add_help=add_help,
    )
    parser.add_argument(
        "--model_dir", required=True, type=str, help="root path to hunyuanDit-v1.2-diffusers models"
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
        "--negative_prompt",
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
        "--guidance_scale", default=5, type=float, help="guidance_scale or CFG scale"
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
    dtype=torch.float16
    if device == "gcu":
        import torch_gcu
    if device == 'cpu':
        dtype=torch.float32

    pipe = HunyuanDiTPipeline.from_pretrained(model_dir, torch_dtype=dtype)
    pipe = pipe.to(device)
    return pipe

def run_hunyuanDiT_v1_2_diffusers_text2image(
                          pipe,
                          image_height,
                          image_width,
                          seed,
                          prompt_list,
                          negative_prompt_list,
                          guidance_scale,
                          denoising_steps,
                          num_images_per_prompt,
                          output_dir
                          ):
    if seed is None:
        seed = 42
    os.makedirs(output_dir, exist_ok=True)

    negative_prompt_list, = prompt_align(prompt_list, [negative_prompt_list,])
    images = pipe(
                prompt=prompt_list,
                negative_prompt=negative_prompt_list,
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
    run_hunyuanDiT_v1_2_diffusers_text2image(
                            pipe,
                            args.image_height,
                            args.image_width,
                            args.seed,
                            args.prompt,
                            args.negative_prompt,
                            args.guidance_scale,
                            args.denoising_steps,
                            args.num_images_per_prompt,
                            args.output_dir
                            )
