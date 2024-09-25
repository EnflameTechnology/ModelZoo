import argparse
import json
import os
from glob import glob

from mmengine.config import Config


def parse_args(training=False):
    parser = argparse.ArgumentParser()


    # ======================================================
    # General
    # ======================================================
    parser.add_argument("--seed", default=42, type=int, help="seed for reproducibility")
    parser.add_argument(
        "--ckpt-path-dit",
        default=None,
        type=str,
        help="path to dit model ckpt; will overwrite cfg.model.from_pretrained if specified",
    )
    parser.add_argument(
        "--ckpt-path-t5",
        default=None,
        type=str,
        help="path to t5 model ckpt; will overwrite cfg.text_encoder.from_pretrained if specified",
    )
    parser.add_argument(
        "--ckpt-path-vae2d",
        default=None,
        type=str,
        help="path to vae2d model ckpt; will overwrite cfg.vae.from_pretrained_vae2d if specified",
    )
    parser.add_argument(
        "--ckpt-path-vae3d",
        default=None,
        type=str,
        help="path to vae3d model ckpt; will overwrite cfg.vae.from_pretrained_vae3d if specified",
    )
    parser.add_argument(
        "--warmup-count",
        default=1,
        type=int,
        help="times of inference execution are needed for preheating/warmup",
    )
    parser.add_argument(
        "--eval-count",
        default=1,
        type=int,
        help="times of inference execution are needed for performance evaluation",
    )
    parser.add_argument(
        "--benchmark-save-path",
        default='./benchmark/report-opensora-v1.2.json',
        type=str,
        help="a json file path to save benchmark-test result, e.g: ./benchmark/report-opensora-v1.2.json"
    )

    parser.add_argument("--batch-size", default=1, type=int, help="batch size")
    parser.add_argument("--model-max-length", default=300, type=int, help="t5 model max length")
    parser.add_argument("--micro-batch-size", default=4, type=int, help="vae micro batch size")
    parser.add_argument("--micro-frame-size", default=17, type=int, help="vae micro frame size")
    parser.add_argument("--outputs", default=None, type=str, help="the dir to save model weights")
    parser.add_argument("--flash-attn", default=False, type=str2bool, help="enable flash attention")
    parser.add_argument("--layernorm-kernel", default=False, type=str2bool, help="enable layernorm kernel")
    parser.add_argument("--enable-compile", default=False, type=str2bool, help="enable torch compile")
    parser.add_argument("--qk-norm", default=True, type=str2bool, help="enable qk_norm")
    parser.add_argument("--resolution", default="240p", type=str, help="multi resolution")
    parser.add_argument("--data-path", default=None, type=str, help="path to data csv")
    parser.add_argument("--dtype", default="bf16", type=str, help="data type")
    parser.add_argument("--multi_resolution", default="STDiT2", type=str, help="multi_resolution")
    parser.add_argument("--device", default='gcu', type=str, choices=['cpu', 'cuda', 'gcu'],
        help="Which device do you want to run the program on, CPU, GPU, or GCU?")
    parser.add_argument("--align", default=5, type=int, help="align")
    # ======================================================
    # Inference
    # ======================================================
    if not training:
        # output
        parser.add_argument("--save-dir", default="./samples/samples/", type=str, help="path to save generated samples")
        parser.add_argument("--sample-name", default=None, type=str, help="sample name, default is sample_idx")
        parser.add_argument("--start-index", default=None, type=int, help="start index for sample name")
        parser.add_argument("--end-index", default=None, type=int, help="end index for sample name")
        parser.add_argument("--num-sample", default=None, type=int, help="number of samples to generate for one prompt")
        parser.add_argument("--prompt-as-path", action="store_true", help="use prompt as path to save samples")
        parser.add_argument("--verbose", default=None, type=int, help="verbose level")

        # prompt
        parser.add_argument("--prompt-path", default=None, type=str, help="path to prompt txt file")
        parser.add_argument("--prompt", default=None, type=str, nargs="+", help="prompt list")
        parser.add_argument("--llm-refine", default=None, type=str2bool, help="enable LLM refine")
        parser.add_argument("--prompt-generator", default=None, type=str, help="prompt generator")

        # image/video
        parser.add_argument("--num-frames", default=51, type=str, help="number of frames")
        parser.add_argument("--fps", default=24, type=int, help="fps")
        parser.add_argument("--save-fps", default=24, type=int, help="save fps")
        parser.add_argument("--image-size", default=None, type=int, nargs=2, help="image size")
        parser.add_argument("--frame-interval", default=1, type=int, help="frame interval")
        parser.add_argument("--aspect-ratio", default="9:16", type=str, help="aspect ratio (h:w)")
        parser.add_argument("--watermark", default=None, type=str2bool, help="watermark video")

        # hyperparameters
        parser.add_argument("--num-sampling-steps", default=30, type=int, help="sampling steps")
        parser.add_argument("--cfg-scale", default=7.0, type=float, help="balance between cond & uncond")

        # reference
        parser.add_argument("--loop", default=None, type=int, help="loop")
        parser.add_argument("--condition-frame-length", default=5, type=int, help="condition frame length")
        parser.add_argument("--reference-path", default=None, type=str, nargs="+", help="reference path")
        parser.add_argument("--mask-strategy", default=None, type=str, nargs="+", help="mask strategy")
        parser.add_argument("--aes", default=6.5, type=float, help="aesthetic score")
        parser.add_argument("--flow", default=None, type=float, help="flow score")
        parser.add_argument("--camera-motion", default=None, type=str, help="camera motion")
    # ======================================================
    # Training
    # ======================================================
    else:
        parser.add_argument("--lr", default=None, type=float, help="learning rate")
        parser.add_argument("--wandb", default=None, type=bool, help="enable wandb")
        parser.add_argument("--load", default=None, type=str, help="path to continue training")
        parser.add_argument("--start-from-scratch", action="store_true", help="start training from scratch")
        parser.add_argument("--warmup-steps", default=None, type=int, help="warmup steps")

    return parser.parse_args()


def merge_args(cfg, args, training=False):
    if args.ckpt_path_dit is not None:
        cfg.model["from_pretrained"] = args.ckpt_path_dit
        if cfg.get("discriminator") is not None:
            cfg.discriminator["from_pretrained"] = args.ckpt_path_dit
        args.ckpt_path_dit = None
    if args.ckpt_path_vae2d is not None:
        cfg.vae["from_pretrained_vae2d"] = args.ckpt_path_vae2d
        args.ckpt_path_vae2d = None
    if args.ckpt_path_vae3d is not None:
        cfg.vae["from_pretrained_vae3d"] = args.ckpt_path_vae3d
        args.ckpt_path_vae3d = None
    if args.ckpt_path_t5 is not None:
        cfg.text_encoder["from_pretrained"] = args.ckpt_path_t5
        args.ckpt_path_t5 = None
    if args.model_max_length is not None:
        cfg.text_encoder["model_max_length"] = args.model_max_length
        args.model_max_length = None
    if args.micro_batch_size is not None:
        cfg.vae["micro_batch_size"] = args.micro_batch_size
        args.micro_batch_size = None
    if args.micro_frame_size is not None:
        cfg.vae["micro_frame_size"] = args.micro_frame_size
        args.micro_frame_size = None
    if args.flash_attn is not None:
        cfg.model["enable_flash_attn"] = args.flash_attn
        args.enable_flash_attn = None
    if args.device is not None:
        cfg.device = args.device
        args.device = None
    if args.layernorm_kernel is not None:
        cfg.model["enable_layernorm_kernel"] = args.layernorm_kernel
        args.enable_layernorm_kernel = None
    if args.enable_compile is not None:
        cfg.model["enable_compile"] = args.enable_compile
        args.enable_compile = None
    if args.qk_norm is not None:
        cfg.model["qk_norm"] = args.qk_norm
        args.qk_norm = None
    if args.data_path is not None:
        cfg.dataset["data_path"] = args.data_path
        args.data_path = None
    # NOTE: for vae inference (reconstruction)
    if not training and "dataset" in cfg:
        if args.image_size is not None:
            cfg.dataset["image_size"] = args.image_size
        if args.num_frames is not None:
            cfg.dataset["num_frames"] = args.num_frames
    if not training:
        if args.cfg_scale is not None:
            cfg.scheduler["cfg_scale"] = args.cfg_scale
            args.cfg_scale = None
        if args.num_sampling_steps is not None:
            cfg.scheduler["num_sampling_steps"] = args.num_sampling_steps
            args.num_sampling_steps = None

    for k, v in vars(args).items():
        if v is not None:
            cfg[k] = v

    return cfg


def read_config(config_path):
    cfg = Config.fromfile(config_path)
    return cfg


def parse_configs(training=False):
    args = parse_args(training)
    cfg = Config(dict(
        model=dict(type='STDiT3-XL/2',
                   from_pretrained=''),
        vae=dict(type='OpenSoraVAE_V1_2',
                 from_pretrained_vae2d='',
                 from_pretrained_vae3d=''),
        text_encoder=dict(type='t5',
                          from_pretrained=''),
        scheduler=dict(type='rflow'),
        ))
    cfg = merge_args(cfg, args, training)
    return cfg


def define_experiment_workspace(cfg, get_last_workspace=False):
    """
    This function creates a folder for experiment tracking.

    Args:
        args: The parsed arguments.

    Returns:
        exp_dir: The path to the experiment folder.
    """
    # Make outputs folder (holds all experiment subfolders)
    os.makedirs(cfg.outputs, exist_ok=True)
    experiment_index = len(glob(f"{cfg.outputs}/*"))
    if get_last_workspace:
        experiment_index -= 1

    # Create an experiment folder
    model_name = cfg.model["type"].replace("/", "-")
    exp_name = f"{experiment_index:03d}-{model_name}"
    exp_dir = f"{cfg.outputs}/{exp_name}"
    return exp_name, exp_dir


def save_training_config(cfg, experiment_dir):
    with open(f"{experiment_dir}/config.txt", "w") as f:
        json.dump(cfg, f, indent=4)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")
