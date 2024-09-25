import os
import json
import time
from pprint import pformat

import torch
import torch.distributed as dist
from mmengine.runner import set_random_seed
from tqdm import tqdm

import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)
from common.get_meta_info import get_meta_info
from open_sora.opensora.acceleration.parallel_states import set_sequence_parallel_group
from open_sora.opensora.datasets import save_sample
from open_sora.opensora.datasets.aspect import get_image_size, get_num_frames
from open_sora.opensora.models.text_encoder.t5 import text_preprocessing, T5Encoder
from open_sora.opensora.models.vae import OpenSoraVAE_V1_2
from open_sora.opensora.models.stdit import STDiT3_XL_2
from open_sora.opensora.schedulers.rf import RFLOW
from open_sora.opensora.registry import MODELS, SCHEDULERS, build_module
from open_sora.opensora.utils.config_utils import parse_configs
from open_sora.opensora.utils.inference_utils import (
    add_watermark,
    append_generated,
    append_score_to_prompts,
    apply_mask_strategy,
    collect_references_batch,
    dframe_to_frame,
    extract_json_from_prompts,
    extract_prompts_loop,
    get_save_path_name,
    load_prompts,
    merge_prompt,
    prepare_multi_resolution_info,
    refine_prompts_by_openai,
    split_prompt,
)
from open_sora.opensora.utils.misc import all_exists, create_logger, is_distributed, is_main_process, to_torch_dtype

def saveVideo(batch_prompts, verbose, logger, save_paths, video_clips, loop, condition_frame_length, save_fps, cfg):
    if is_main_process():
        for idx, batch_prompt in enumerate(batch_prompts):
            if verbose >= 2:
                logger.info("Prompt: %s", batch_prompt)
            save_path = save_paths[idx]
            video = [video_clips[i][idx] for i in range(loop)]
            for i in range(1, loop):
                video[i] = video[i][:, dframe_to_frame(condition_frame_length) :]
            video = torch.cat(video, dim=1)
            save_path = save_sample(
                video,
                fps=save_fps,
                save_path=save_path,
                verbose=verbose >= 2,
            )
            if save_path.endswith(".mp4") and cfg.get("watermark", False):
                time.sleep(1)  # prevent loading previous generated video
                add_watermark(save_path)

def get_device_time(device):
    if device == "cuda" :
        torch.cuda.synchronize()
    elif device == "gcu":
        torch.gcu.synchronize()
    return time.time()

def run_video(cfg, text_encoder, model, vae, num_sampling_steps, progress_wrap, prompts, batch_size, warm_up, save_dir,
         logger, mask_strategy, reference_path, image_size, multi_resolution, num_frames, fps, device,
         dtype, num_sample, sample_name, prompt_as_path, enable_sequence_parallelism, coordinator, loop,
         condition_frame_length, condition_frame_edit, latent_size, align, verbose, start_idx, save_fps):
    # == build scheduler ==
    scheduler = RFLOW(
        num_sampling_steps=num_sampling_steps,
        cfg_scale=cfg.scheduler["cfg_scale"],
        use_timestep_transform=True)
    
    # == Iter over all samples ==
    for i in progress_wrap(range(0, len(prompts), batch_size)):
        # == prepare batch prompts ==
        batch_prompts = prompts[i : i + batch_size]
        ms = mask_strategy[i : i + batch_size]
        refs = reference_path[i : i + batch_size]

        # == get json from prompts ==
        batch_prompts, refs, ms = extract_json_from_prompts(batch_prompts, refs, ms)
        original_batch_prompts = batch_prompts

        # == get reference for condition ==
        refs = collect_references_batch(refs, vae, image_size)

        # == multi-resolution info ==
        model_args = prepare_multi_resolution_info(
            multi_resolution, len(batch_prompts), image_size, num_frames, fps, device, dtype
        )

        # == Iter over number of sampling for one prompt ==
        t_infer_start = get_device_time(device)
        all_save_time = 0
        for k in range(num_sample):
            # == prepare save paths ==
            save_paths = [
                get_save_path_name(
                    save_dir,
                    sample_name=sample_name,
                    sample_idx=start_idx + idx,
                    prompt=original_batch_prompts[idx],
                    prompt_as_path=prompt_as_path,
                    num_sample=num_sample,
                    k=k,
                )
                for idx in range(len(batch_prompts))
            ]

            # NOTE: Skip if the sample already exists
            # This is useful for resuming sampling VBench
            if prompt_as_path and all_exists(save_paths):
                continue

            # == process prompts step by step ==
            # 0. split prompt
            # each element in the list is [prompt_segment_list, loop_idx_list]
            batched_prompt_segment_list = []
            batched_loop_idx_list = []
            for prompt in batch_prompts:
                prompt_segment_list, loop_idx_list = split_prompt(prompt)
                batched_prompt_segment_list.append(prompt_segment_list)
                batched_loop_idx_list.append(loop_idx_list)

            # 1. refine prompt by openai
            if cfg.get("llm_refine", False):
                # only call openai API when
                # 1. seq parallel is not enabled
                # 2. seq parallel is enabled and the process is rank 0
                if not enable_sequence_parallelism or (enable_sequence_parallelism and is_main_process()):
                    for idx, prompt_segment_list in enumerate(batched_prompt_segment_list):
                        batched_prompt_segment_list[idx] = refine_prompts_by_openai(prompt_segment_list)

                # sync the prompt if using seq parallel
                if enable_sequence_parallelism:
                    coordinator.block_all()
                    prompt_segment_length = [
                        len(prompt_segment_list) for prompt_segment_list in batched_prompt_segment_list
                    ]

                    # flatten the prompt segment list
                    batched_prompt_segment_list = [
                        prompt_segment
                        for prompt_segment_list in batched_prompt_segment_list
                        for prompt_segment in prompt_segment_list
                    ]

                    # create a list of size equal to world size
                    broadcast_obj_list = [batched_prompt_segment_list] * coordinator.world_size
                    dist.broadcast_object_list(broadcast_obj_list, 0)

                    # recover the prompt list
                    batched_prompt_segment_list = []
                    segment_start_idx = 0
                    all_prompts = broadcast_obj_list[0]
                    for num_segment in prompt_segment_length:
                        batched_prompt_segment_list.append(
                            all_prompts[segment_start_idx : segment_start_idx + num_segment]
                        )
                        segment_start_idx += num_segment

            # 2. append score
            for idx, prompt_segment_list in enumerate(batched_prompt_segment_list):
                batched_prompt_segment_list[idx] = append_score_to_prompts(
                    prompt_segment_list,
                    aes=cfg.get("aes", None),
                    flow=cfg.get("flow", None),
                    camera_motion=cfg.get("camera_motion", None),
                )

            # 3. clean prompt with T5
            for idx, prompt_segment_list in enumerate(batched_prompt_segment_list):
                batched_prompt_segment_list[idx] = [text_preprocessing(prompt) for prompt in prompt_segment_list]

            # 4. merge to obtain the final prompt
            batch_prompts = []
            for prompt_segment_list, loop_idx_list in zip(batched_prompt_segment_list, batched_loop_idx_list):
                batch_prompts.append(merge_prompt(prompt_segment_list, loop_idx_list))

            # == Iter over loop generation ==
            video_clips = []
            for loop_i in range(loop):
                # == get prompt for loop i ==
                batch_prompts_loop = extract_prompts_loop(batch_prompts, loop_i)
                
                # == add condition frames for loop ==
                if loop_i > 0:
                    refs, ms = append_generated(
                        vae, video_clips[-1], refs, ms, loop_i, condition_frame_length, condition_frame_edit
                    )

                # == sampling ==
                z = torch.randn(len(batch_prompts), vae.out_channels, *latent_size, device='cpu', dtype=dtype).to(device)
                masks = apply_mask_strategy(z, refs, ms, loop_i, align=align)
                
                samples = scheduler.sample(
                    model,
                    text_encoder,
                    z=z,
                    prompts=batch_prompts_loop,
                    device=device,
                    additional_args=model_args,
                    progress=verbose >= 2,
                    mask=masks,
                )
                print(f"###STDIT end###")
                print(f'STDIT samples.shape: {samples.shape}, dtype: {samples.dtype}')
                samples = vae.decode(samples.to(dtype), num_frames=num_frames)
                print(f'VAE samples.shape: {samples.shape}, dtype: {samples.dtype}')
                print(f"###VAE end###")
                video_clips.append(samples)
                
            # == save samples ==
            t_save_start = get_device_time(device)
            if not warm_up:
                saveVideo(batch_prompts, verbose, logger, save_paths, video_clips, loop, condition_frame_length, save_fps, cfg)
            t_save_end = get_device_time(device)
            all_save_time += (t_save_end - t_save_start)

        start_idx += len(batch_prompts)
        t_infer_end = get_device_time(device)
        avg_single_video_save_time = all_save_time / num_sample
        avg_single_infer_time = (t_infer_end - t_infer_start - all_save_time) / (loop * num_sample)  
        if not warm_up:
            print(f'Avg single prompt infer time: {avg_single_infer_time} s')
            print(f'Avg single video save time: {avg_single_video_save_time} s')
    if not warm_up:
        logger.info("Inference finished.")
        logger.info("Saved %s samples to %s", start_idx, save_dir)

    return avg_single_infer_time, avg_single_video_save_time

def videogen_pipeline(cfg, text_encoder, model, vae, image_size, num_frames, device, dtype, enable_sequence_parallelism, coordinator, latent_size, logger, load_time):
    verbose = cfg.get("verbose", 1)
    progress_wrap = tqdm if verbose == 1 else (lambda x: x)
    # ======================================================
    # inference
    # ======================================================
    # == load prompts ==
    prompts = cfg.get("prompt", None)
    start_idx = cfg.get("start_index", 0)
    if prompts is None:
        if cfg.get("prompt_path", None) is not None:
            prompts = load_prompts(cfg.prompt_path, start_idx, cfg.get("end_index", None))
        else:
            prompts = [cfg.get("prompt_generator", "")] * 1_000_000  # endless loop

    # == prepare reference ==
    reference_path = cfg.get("reference_path", [""] * len(prompts))
    mask_strategy = cfg.get("mask_strategy", [""] * len(prompts))
    assert len(reference_path) == len(prompts), "Length of reference must be the same as prompts"
    assert len(mask_strategy) == len(prompts), "Length of mask_strategy must be the same as prompts"

    # == prepare arguments ==
    fps = cfg.fps
    save_fps = cfg.get("save_fps", fps // cfg.get("frame_interval", 1))
    multi_resolution = cfg.get("multi_resolution", None)
    batch_size = cfg.get("batch_size", 1)
    num_sample = cfg.get("num_sample", 1)
    loop = cfg.get("loop", 1)
    condition_frame_length = cfg.get("condition_frame_length", 5)
    condition_frame_edit = cfg.get("condition_frame_edit", 0.0)
    align = cfg.get("align", None)

    save_dir = cfg.save_dir
    os.makedirs(save_dir, exist_ok=True)
    sample_name = cfg.get("sample_name", None)
    prompt_as_path = cfg.get("prompt_as_path", False)

    # warm up
    print("Warming up start...")
    for m in range(cfg.get("warmup_count", 1)):
        print(f'Warmp up {m} time...')
        _, _ = run_video(cfg, text_encoder, model, vae, 5, progress_wrap, prompts, batch_size, True, save_dir,
         logger, mask_strategy, reference_path, image_size, multi_resolution, num_frames, fps, device,
         dtype, num_sample, sample_name, prompt_as_path, enable_sequence_parallelism, coordinator, loop,
         condition_frame_length, condition_frame_edit, latent_size, align, verbose, start_idx, save_fps)
    print("Warming up end...")

    all_infer_time = 0
    all_save_time = 0
    tic = get_device_time(device)
    for m in range(cfg.get("eval_count", 1)):
        print(f' Eval video pipeline {m} time...')
        infer_time, save_time = run_video(cfg, text_encoder, model, vae, cfg.scheduler["num_sampling_steps"],
            progress_wrap, prompts, batch_size, False, save_dir,
            logger, mask_strategy, reference_path, image_size, multi_resolution, num_frames, fps, device,
            dtype, num_sample, sample_name, prompt_as_path, enable_sequence_parallelism, coordinator, loop,
            condition_frame_length, condition_frame_edit, latent_size, align, verbose, start_idx, save_fps)
        all_save_time = all_save_time + save_time
        all_infer_time = all_infer_time + infer_time

    tid = get_device_time(device)
    print(f'\n{"-" * 50}\ninference time is: {(tid - tic) / cfg.get("eval_count", 1):.2f} s\n{"-" * 50}\n')
    QPS = num_sample * num_frames * 1000 / ((tid - tic) / cfg.get("eval_count", 1))
    print(f"QPS: {QPS:.2f} frames/s")
    
    performance_info_dict = {
        'QPS': QPS,
        'model_load_time': load_time,
        'resolution':cfg.get("resolution", '360p'),
        'aspect_ratio':cfg.get("aspect_ratio", '9:16'),
        'batch_size':cfg.get("batch_size", 1),
        'dtype':cfg.get("dtype", 'bf16'),
        'scheduler': cfg.scheduler["type"],
        'fps':fps,
        'num_frames':num_frames,
        'num_sample':num_sample,
        'denoising_steps':cfg.scheduler["num_sampling_steps"],
        'seed': cfg.get("seed", 1024),
        'device': device,
        'output_path': save_dir,
        'warmup_count': cfg.get("warmup_count", 1),
        'eval_count': cfg.get("eval_count", 1),
        'Avg infer time': (tid - tic) / cfg.get("eval_count", 1),
        'Avg single prompt infer time': all_infer_time / cfg.get("eval_count", 1),
        'Avg single video save time': all_save_time / cfg.get("eval_count", 1),
    }

    meta_info = get_meta_info()
    info_dict = {'performance_info': performance_info_dict}
    info_dict.update(meta_info)
    benchmark_json_path = cfg.get("benchmark_save_path", '')
    os.makedirs(os.path.dirname(benchmark_json_path), exist_ok=True)
    with open(benchmark_json_path, "w") as f:
        f.write(json.dumps(info_dict, indent=4))
    print(f'performance data have been written to {benchmark_json_path}')

def main():
    torch.set_grad_enabled(False)
    # ======================================================
    # configs & runtime variables
    # ======================================================
    # == parse configs ==
    cfg = parse_configs(training=False)

    # == device and dtype ==
    device = cfg.get("device", "cuda")
    if device == "gcu":
        import torch_gcu
    all_eval_time_s = get_device_time(device)
    cfg_dtype = cfg.get("dtype", "fp32")
    assert cfg_dtype in ["fp16", "bf16", "fp32"], f"Unknown mixed precision {cfg_dtype}"
    dtype = to_torch_dtype(cfg.get("dtype", "bf16"))

    # == init distributed env ==
    if is_distributed():
        pass
    else:
        coordinator = None
        enable_sequence_parallelism = False
    set_random_seed(seed=cfg.get("seed", 1024))

    # == init logger ==
    logger = create_logger()
    logger.info("Inference configuration:\n %s", pformat(cfg.to_dict()))

    # ======================================================
    # build model & load weights
    # ======================================================
    logger.info("Building models...")
    time_s_load = get_device_time(device)
    # == build text-encoder and vae ==
    text_encoder = T5Encoder(
        from_pretrained=cfg.text_encoder["from_pretrained"],
        model_max_length=cfg.text_encoder["model_max_length"],
        device=device,
        dtype=torch.bfloat16,
        cache_dir=None,
        shardformer=False,
        local_files_only=False
    )

    vae = OpenSoraVAE_V1_2(
            micro_batch_size=cfg.vae["micro_batch_size"],
            micro_frame_size=cfg.vae["micro_frame_size"],
            from_pretrained_vae2d=cfg.vae["from_pretrained_vae2d"],
            from_pretrained_vae3d=cfg.vae["from_pretrained_vae3d"],
            local_files_only=False,
            freeze_vae_2d=False,
            cal_loss=False,
        ).to(device, dtype).eval()
    # == prepare video size ==
    image_size = cfg.get("image_size", None)
    if image_size is None:
        resolution = cfg.get("resolution", None)
        aspect_ratio = cfg.get("aspect_ratio", None)
        assert (
            resolution is not None and aspect_ratio is not None
        ), "resolution and aspect_ratio must be provided if image_size is not provided"
        image_size = get_image_size(resolution, aspect_ratio)
    num_frames = get_num_frames(cfg.num_frames)

    # == build diffusion model ==
    input_size = (num_frames, *image_size)
    latent_size = vae.get_latent_size(input_size)
    model = STDiT3_XL_2(
        from_pretrained=cfg.model["from_pretrained"],
        qk_norm=cfg.model["qk_norm"],
        enable_flash_attn=cfg.model["enable_layernorm_kernel"],
        enable_layernorm_kernel=cfg.model["from_pretrained"],
        input_size=latent_size,
        in_channels=vae.out_channels,
        caption_channels=text_encoder.output_dim,
        model_max_length=text_encoder.model_max_length,
        enable_sequence_parallelism=enable_sequence_parallelism,
    ).to(device, dtype).eval()

    time_e_load = get_device_time(device) - time_s_load
    print(f'\n{"-" * 50}\nloading model time is: {time_e_load:.2f} s\n{"-" * 50}\n')
    
    if device == 'gcu' and cfg.model["enable_compile"]:
        options = {"subgraph_mode": True}
        model = torch.compile(model, backend="topsgraph", options=options)
    elif device == 'cuda' and cfg.model["enable_compile"]:
        model = torch.compile(model)
    text_encoder.y_embedder = model.y_embedder  # HACK: for classifier-free guidance

    videogen_pipeline(cfg, text_encoder, model, vae, image_size, num_frames, device, dtype,
                      enable_sequence_parallelism, coordinator, latent_size, logger, time_e_load)
    all_eval_time_e = get_device_time(device) - all_eval_time_s
    print(f'total evaluation time: {all_eval_time_e}')
    print('task finished')


if __name__ == "__main__":
    main()