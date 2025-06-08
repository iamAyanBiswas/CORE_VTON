import argparse
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

from model.cloth_masker import AutoMasker, vis_mask
from model.pipeline import CatVTONPipeline
from utils import resize_and_crop, resize_and_padding

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", type=str, default="booksforcharlie/stable-diffusion-inpainting")
    parser.add_argument("--p2p_base_model_path", type=str, default="timbrooks/instruct-pix2pix")
    parser.add_argument("--resume_path", type=str, default="zhengchong/CatVTON")
    parser.add_argument("--output_dir", type=str, default="resource/demo/output")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--repaint", action="store_true")
    parser.add_argument("--allow_tf32", action="store_true", default=True)
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    return parser.parse_args()

def image_grid(imgs, rows, cols):
    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid

args = parse_args()

repo_path = snapshot_download(repo_id=args.resume_path)

pipeline = CatVTONPipeline(
    base_ckpt=args.base_model_path,
    attn_ckpt=repo_path,
    attn_ckpt_version="mix",
    weight_dtype=torch.float16,
    use_tf32=args.allow_tf32,
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
    person_image, mask = person_image["background"], person_image["layers"][0]
    mask = Image.open(mask).convert("L")
    if len(np.unique(np.array(mask))) == 1:
        mask = None
    else:
        mask = np.array(mask)
        mask[mask > 0] = 255
        mask = Image.fromarray(mask)

    tmp_folder = args.output_dir
    date_str = datetime.now().strftime("%Y%m%d%H%M%S")
    result_save_path = os.path.join(tmp_folder, date_str[:8], date_str[8:] + ".png")
    os.makedirs(os.path.dirname(result_save_path), exist_ok=True)

    generator = torch.Generator(device='cuda').manual_seed(seed) if seed != -1 else None

    person_image = resize_and_crop(Image.open(person_image).convert("RGB"), (args.width, args.height))
    cloth_image = resize_and_padding(Image.open(cloth_image).convert("RGB"), (args.width, args.height))

    if mask is not None:
        mask = resize_and_crop(mask, (args.width, args.height))
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

def person_example_fn(image_path):
    return image_path

HEADER = """
<h1 style="text-align: center;"> 🐈 CatVTON: Concatenation Is All You Need for Virtual Try-On with Diffusion Models </h1>
"""

def app_gradio():
    with gr.Blocks(title="CatVTON") as demo:
        gr.Markdown(HEADER)
        with gr.Tab("Mask-based & SD1.5"):
            with gr.Row():
                with gr.Column(scale=1, min_width=350):
                    with gr.Row():
                        image_path = gr.Image(type="filepath", interactive=True, visible=False)
                        person_image = gr.ImageEditor(interactive=True, label="Person Image", type="filepath")
                    with gr.Row():
                        with gr.Column(scale=1, min_width=230):
                            cloth_image = gr.Image(interactive=True, label="Condition Image", type="filepath")
                        with gr.Column(scale=1, min_width=120):
                            cloth_type = gr.Radio(label="Try-On Cloth Type", choices=["upper", "lower", "overall"], value="upper")
                    submit = gr.Button("Submit")
                    with gr.Accordion("Advanced Options", open=False):
                        num_inference_steps = gr.Slider(label="Inference Step", minimum=10, maximum=100, step=5, value=30)
                        guidance_scale = gr.Slider(label="CFG Strength", minimum=0.0, maximum=7.5, step=0.5, value=2.5)
                        seed = gr.Slider(label="Seed", minimum=-1, maximum=10000, step=1, value=42)
                        show_type = gr.Radio(label="Show Type", choices=["result only", "input & result", "input & mask & result"], value="input & mask & result")
                with gr.Column(scale=2, min_width=500):
                    result_image = gr.Image(interactive=False, label="Result")
                    root_path = "resource/demo/example"
                    with gr.Row():
                        with gr.Column():
                            men_exm = gr.Examples([
                                os.path.join(root_path, "person", "men", _) for _ in os.listdir(os.path.join(root_path, "person", "men"))
                            ], examples_per_page=4, inputs=image_path, label="Person Examples ①")
                            women_exm = gr.Examples([
                                os.path.join(root_path, "person", "women", _) for _ in os.listdir(os.path.join(root_path, "person", "women"))
                            ], examples_per_page=4, inputs=image_path, label="Person Examples ②")
                        with gr.Column():
                            condition_upper_exm = gr.Examples([
                                os.path.join(root_path, "condition", "upper", _) for _ in os.listdir(os.path.join(root_path, "condition", "upper"))
                            ], examples_per_page=4, inputs=cloth_image, label="Condition Upper Examples")
                            condition_overall_exm = gr.Examples([
                                os.path.join(root_path, "condition", "overall", _) for _ in os.listdir(os.path.join(root_path, "condition", "overall"))
                            ], examples_per_page=4, inputs=cloth_image, label="Condition Overall Examples")
                image_path.change(person_example_fn, inputs=image_path, outputs=person_image)
                submit.click(submit_function, [person_image, cloth_image, cloth_type, num_inference_steps, guidance_scale, seed, show_type], result_image)
    demo.queue().launch(share=True, show_error=True)

if __name__ == "__main__":
    app_gradio()
