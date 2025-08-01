import os
os.environ['CUDA_HOME'] = '/usr/local/cuda'
os.environ['PATH'] += ':/usr/local/cuda/bin'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from datetime import datetime

import gradio as gr
import spaces
import numpy as np
import torch
from diffusers.image_processor import VaeImageProcessor
from huggingface_hub import snapshot_download
from PIL import Image

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.jit.script = lambda f: f

from vton_model.model.cloth_masker import AutoMasker, vis_mask
from vton_model.model.pipeline import CatVTONPipeline

from vton_model.utils import resize_and_crop, resize_and_padding


def image_grid(imgs, rows, cols):
    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid

args ={
    'base_model_path':'booksforcharlie/stable-diffusion-inpainting',
    'resume_path':'zhengchong/CatVTON',
    'output_dir':'resource/demo/output',
    'width':512,
    'height':768,
    'allow_tf32':True,
    'mixed_precision':'fp16'
}

repo_path = snapshot_download(repo_id=args['resume_path'])

pipeline = CatVTONPipeline(
    base_ckpt=args['base_model_path'],
    attn_ckpt=repo_path,
    attn_ckpt_version="mix",
    weight_dtype=torch.float16,
    use_tf32=args['allow_tf32'],
    device='cuda'
)


mask_processor = VaeImageProcessor(vae_scale_factor=8, do_normalize=False, do_binarize=True, do_convert_grayscale=True)
automasker = AutoMasker(
    densepose_ckpt=os.path.join(repo_path, "DensePose"),
    schp_ckpt=os.path.join(repo_path, "SCHP"),
    device='cuda',
)





@spaces.GPU(duration=120)
def submit_function(person_image, cloth_image, cloth_type, num_inference_steps, guidance_scale, seed, show_type):
    print({'cloth_type': cloth_type, 'num_inference_steps': num_inference_steps, 'guidance_scale': guidance_scale, 'seed': seed, 'show_type': show_type})

    mask = None

    tmp_folder = args['output_dir']
    date_str = datetime.now().strftime("%Y%m%d%H%M%S")
    result_save_path = os.path.join(tmp_folder, date_str[:8], date_str[8:] + ".png")
    os.makedirs(os.path.dirname(result_save_path), exist_ok=True)

    generator = torch.Generator(device='cuda').manual_seed(seed) if seed != -1 else None

    person_image = resize_and_crop(Image.open(person_image).convert("RGB"), (args['width'], args['height']))
    cloth_image = resize_and_padding(Image.open(cloth_image).convert("RGB"), (args['width'], args['height']))

    if mask is not None:
        mask = resize_and_crop(mask, (args['width'], args['height']))
    else:
        mask = automasker(person_image, cloth_type)['mask']
    mask = mask_processor.blur(mask, blur_factor=9)

    result_image = pipeline(
        image=person_image,
        condition_image=cloth_image,
        mask=mask,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator
    )[0]

    masked_person = vis_mask(person_image, mask)
    save_result_image = image_grid([person_image, masked_person, cloth_image, result_image], 1, 4)
    save_result_image.save(result_save_path)

    if show_type == "result only":
        return result_image
    else:
        width, height = person_image.size
        if show_type == "input & result":
            condition_width = width // 2
            conditions = image_grid([person_image, cloth_image], 2, 1)
        else:
            condition_width = width // 3
            conditions = image_grid([person_image, masked_person, cloth_image], 3, 1)
        conditions = conditions.resize((condition_width, height), Image.NEAREST)
        new_result_image = Image.new("RGB", (width + condition_width + 5, height))
        new_result_image.paste(conditions, (0, 0))
        new_result_image.paste(result_image, (condition_width + 5, 0))
        return new_result_image


