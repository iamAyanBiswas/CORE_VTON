from vton_model.testcolab import submit_function


import gradio as gr
import os

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
                    root_path = "vton_model/resource/demo/example"
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
