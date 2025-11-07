import argparse
import torch
import cv2
import numpy as np
import os
import os.path as osp

from tools.run_infinity import (
    load_tokenizer,
    load_visual_tokenizer,
    load_transformer,
    gen_one_img,
)
from infinity.utils.dynamic_resolution import dynamic_resolution_h_w, h_div_w_templates


def main():
    torch.cuda.set_device(0)
    device = torch.device("cuda")

    model_path='/gemini/space/jiangpf/models/infinity/infinity_125M_256x256.pth'
    vae_path='/gemini/space/jiangpf/models/infinity/infinity_vae_d16.pth'
    text_encoder_ckpt = '/gemini/space/jiangpf/models/infinity/flan-t5-xl'
    args=argparse.Namespace(
        pn='0.06M',
        model_path=model_path,
        cfg_insertion_layer=0,
        vae_type=16,
        vae_path=vae_path,
        add_lvl_embeding_only_first_block=1,
        use_bit_label=1,
        model_type='infinity_layer12',
        rope2d_each_sa_layer=1,
        rope2d_normalized_by_hw=2,
        use_scale_schedule_embedding=0,
        sampling_per_bits=1,
        text_encoder_ckpt=text_encoder_ckpt,
        text_channels=2048,
        apply_spatial_patchify=0,
        h_div_w_template=1.000,
        use_flex_attn=0,
        cache_dir='/dev/shm',
        checkpoint_type='torch',
        seed=0,
        bf16=1,
        save_file_1='tmp.jpg',
        enable_model_cache=0,
    )

    text_tokenizer, text_encoder = load_tokenizer(t5_path=args.text_encoder_ckpt)
    vae = load_visual_tokenizer(args)
    infinity = load_transformer(vae, args)

    torch.cuda.reset_peak_memory_stats(device=device)
    alloc_before_gen = torch.cuda.memory_allocated(device=device) / (1024**2)

    prompt = """alien spaceship enterprise"""
    cfg = 4
    tau = 1.0
    h_div_w = 1/1
    seed = 99
    enable_positive_prompt = 0

    # Prepare scale schedule
    h_div_w_template_ = h_div_w_templates[np.argmin(np.abs(h_div_w_templates - h_div_w))]
    scale_schedule = dynamic_resolution_h_w[h_div_w_template_][args.pn]['scales']
    scale_schedule = [(1, h, w) for (_, h, w) in scale_schedule]

    import time
    start_time = time.time()
    generated_image = gen_one_img(
        infinity,
        vae,
        text_tokenizer,
        text_encoder,
        prompt,
        g_seed=seed,
        gt_leak=0,
        gt_ls_Bl=None,
        cfg_list=cfg,
        tau_list=tau,
        scale_schedule=scale_schedule,
        cfg_insertion_layer=[args.cfg_insertion_layer],
        vae_type=args.vae_type,
        sampling_per_bits=args.sampling_per_bits,
        enable_positive_prompt=enable_positive_prompt,
    )
    end_time = time.time()
    print(f"Generation time: {end_time - start_time:.1f} seconds")

    alloc_after_gen = torch.cuda.memory_allocated(device=device) / (1024**2)
    peak_alloc_gen = torch.cuda.max_memory_allocated(device=device) / (1024**2)

    print("=== Generation Time / Memory Usage of Original Model===")
    print(f"GPU allocated before/after: {alloc_before_gen:.1f} MB -> {alloc_after_gen:.1f} MB (delta {alloc_after_gen - alloc_before_gen:+.1f} MB)")
    print(f"GPU peak allocated during gen: {peak_alloc_gen:.1f} MB (delta {peak_alloc_gen - alloc_before_gen:+.1f} MB)")
    print("======================================\n")

    os.makedirs(osp.dirname(osp.abspath(args.save_file_1)), exist_ok=True)
    cv2.imwrite(args.save_file_1, generated_image.cpu().numpy())
    print(f"Saved streaming result to {args.save_file_1}")


if __name__ == "__main__":
    main()