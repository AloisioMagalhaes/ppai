import os
import numpy as np
import cv2
import kiui
import trimesh
import torch
import rembg
from datetime import datetime
import subprocess
import gradio as gr

try:
    # running on Hugging Face Spaces
    import spaces

except ImportError:
    # running locally, use a dummy space
    class spaces:
        class GPU:
            def __init__(self, duration=60):
                self.duration = duration
            def __call__(self, func):
                return func


from flow.model import Model
from flow.configs.schema import ModelConfig
from flow.utils import get_random_color, recenter_foreground
from vae.utils import postprocess_mesh

# download checkpoints
from huggingface_hub import hf_hub_download
flow_ckpt_path = hf_hub_download(repo_id="nvidia/PartPacker", filename="flow.pt")
vae_ckpt_path = hf_hub_download(repo_id="nvidia/PartPacker", filename="vae.pt")

TRIMESH_GLB_EXPORT = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]]).astype(np.float32)
MAX_SEED = np.iinfo(np.int32).max
bg_remover = rembg.new_session()

# model config
model_config = ModelConfig(
    vae_conf="vae.configs.part_woenc",
    vae_ckpt_path=vae_ckpt_path,
    qknorm=True,
    qknorm_type="RMSNorm",
    use_pos_embed=False,
    dino_model="dinov2_vitg14",
    hidden_dim=1536,
    flow_shift=3.0,
    logitnorm_mean=1.0,
    logitnorm_std=1.0,
    latent_size=4096,
    use_parts=True,
)

# instantiate model
model = Model(model_config).eval().cuda().bfloat16()

# load weight
ckpt_dict = torch.load(flow_ckpt_path, weights_only=True)
model.load_state_dict(ckpt_dict, strict=True)

# get random seed
def get_random_seed(randomize_seed, seed):
    if randomize_seed:
        seed = np.random.randint(0, MAX_SEED)
    return seed

# process image
@spaces.GPU(duration=10)
def process_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image.shape[-1] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # bg removal if there is no alpha channel
        image = rembg.remove(image, session=bg_remover)  # [H, W, 4]
    mask = image[..., -1] > 0
    image = recenter_foreground(image, mask, border_ratio=0.1)
    image = cv2.resize(image, (518, 518), interpolation=cv2.INTER_AREA)
    return image

# process generation
@spaces.GPU(duration=90)
def process_3d(input_image, num_steps=50, cfg_scale=7, grid_res=384, seed=42, simplify_mesh=False, target_num_faces=100000):

    # seed
    kiui.seed_everything(seed)

    # output path
    os.makedirs("output", exist_ok=True)
    output_glb_path = f"output/partpacker_{datetime.now().strftime('%Y%m%d_%H%M%S')}.glb"

    # input image (assume processed to RGBA uint8)
    image = input_image.astype(np.float32) / 255.0
    image = image[..., :3] * image[..., 3:4] + (1 - image[..., 3:4])  # white background
    image_tensor = torch.from_numpy(image).permute(2, 0, 1).contiguous().unsqueeze(0).float().cuda()

    data = {"cond_images": image_tensor}

    with torch.inference_mode():
        results = model(data, num_steps=num_steps, cfg_scale=cfg_scale)

    latent = results["latent"]

    # query mesh

    data_part0 = {"latent": latent[:, : model.config.latent_size, :]}
    data_part1 = {"latent": latent[:, model.config.latent_size :, :]}

    with torch.inference_mode():
        results_part0 = model.vae(data_part0, resolution=grid_res)
        results_part1 = model.vae(data_part1, resolution=grid_res)

    if not simplify_mesh:
        target_num_faces = -1

    vertices, faces = results_part0["meshes"][0]
    mesh_part0 = trimesh.Trimesh(vertices, faces)
    mesh_part0.vertices = mesh_part0.vertices @ TRIMESH_GLB_EXPORT.T
    mesh_part0 = postprocess_mesh(mesh_part0, target_num_faces)
    parts = mesh_part0.split(only_watertight=False)

    vertices, faces = results_part1["meshes"][0]
    mesh_part1 = trimesh.Trimesh(vertices, faces)
    mesh_part1.vertices = mesh_part1.vertices @ TRIMESH_GLB_EXPORT.T
    mesh_part1 = postprocess_mesh(mesh_part1, target_num_faces)
    parts.extend(mesh_part1.split(only_watertight=False))

    # split connected components and assign different colors
    for j, part in enumerate(parts):
        # each component uses a random color
        part.visual.vertex_colors = get_random_color(j, use_float=True)

    mesh = trimesh.Scene(parts)
    # export the whole mesh
    mesh.export(output_glb_path)

    return output_glb_path

# gradio UI

_TITLE = '''🎨 Image to 3D Model - Bring Your Images to Life!'''

_DESCRIPTION = '''
<div style="text-align: center; margin-bottom: 20px;">
    <h3 style="color: #2e7d32;">✨ Transform 2D Images into Stunning 3D Models with One Click ✨</h3>
</div>

### 🚀 Key Features:
- **Smart Recognition**: Automatically identifies objects in images and generates corresponding 3D models
- **Part Separation**: Generated 3D models are automatically decomposed into multiple parts, each displayed in different colors
- **Background Removal**: Automatically removes image backgrounds to ensure only the main object is modeled
- **Universal Format**: Outputs standard GLB format, compatible with various 3D software

### 📖 How to Use:
1. **Upload Image**: Click the "Upload Image" area on the left to upload your picture (supports JPG, PNG, etc.)
2. **Adjust Settings** (Optional):
   - Higher inference steps = better quality but slower (default 50 recommended)
   - If unsatisfied with results, try different random seeds
3. **Click Generate**: Click the "Generate 3D Model" button and wait about 1-2 minutes
4. **View Results**: The 3D model will appear on the right, drag with mouse to rotate and view

### 💡 Tips for Best Results:
- Clear subjects with simple backgrounds work best
- Front-facing or 45-degree angle photos recommended
- If results aren't ideal, try adjusting the random seed and regenerating
- Check the example images below to see optimal input types

### 🎯 Use Cases:
- **Product Display**: Convert product images to 3D models for e-commerce
- **Creative Design**: Quickly obtain 3D prototypes for design reference
- **Game Development**: Generate initial 3D models for game assets
- **Educational Demos**: Convert flat diagrams to 3D for better spatial understanding
'''

block = gr.Blocks(title=_TITLE).queue()

with block:
    with gr.Row():
        with gr.Column():
            gr.Markdown('# ' + _TITLE)
    gr.Markdown(_DESCRIPTION)

    with gr.Row():
        with gr.Column(scale=1):
            with gr.Row():
                # input image
                input_image = gr.Image(
                    label="📷 Upload Image", 
                    type="filepath"
                )
                seg_image = gr.Image(
                    label="🔍 Processed Image", 
                    type="numpy", 
                    interactive=False, 
                    image_mode="RGBA"
                )
            
            with gr.Accordion("⚙️ Advanced Settings", open=False):
                gr.Markdown("""
                ### Parameter Guide:
                - **Inference Steps**: More steps = higher quality but longer processing time
                - **CFG Scale**: Controls generation accuracy, higher values stay closer to original
                - **Grid Resolution**: 3D model detail level, higher = more detailed
                - **Random Seed**: Same seed produces same results, useful for reproducing effects
                - **Simplify Mesh**: Reduces model face count for lightweight applications
                """)
                # inference steps
                num_steps = gr.Slider(
                    label="Inference Steps", 
                    minimum=1, 
                    maximum=100, 
                    step=1, 
                    value=50,
                    info="Recommended: 30-70"
                )
                # cfg scale
                cfg_scale = gr.Slider(
                    label="CFG Scale", 
                    minimum=2, 
                    maximum=10, 
                    step=0.1, 
                    value=7.0,
                    info="Recommended: 6-8"
                )
                # grid resolution
                input_grid_res = gr.Slider(
                    label="Grid Resolution", 
                    minimum=256, 
                    maximum=512, 
                    step=1, 
                    value=384,
                    info="Recommended: 384"
                )
                # random seed
                with gr.Row():
                    randomize_seed = gr.Checkbox(
                        label="Randomize Seed", 
                        value=True,
                        info="Use different seed each time"
                    )
                    seed = gr.Slider(
                        label="Seed Value", 
                        minimum=0, 
                        maximum=MAX_SEED, 
                        step=1, 
                        value=0
                    )
                # simplify mesh
                with gr.Row():
                    simplify_mesh = gr.Checkbox(
                        label="Simplify Mesh", 
                        value=False,
                        info="Reduce model complexity"
                    )
                    target_num_faces = gr.Slider(
                        label="Target Face Count", 
                        minimum=10000, 
                        maximum=1000000, 
                        step=1000, 
                        value=100000,
                        info="Lower count = simpler model"
                    )
                
            # gen button
            button_gen = gr.Button("🎯 Generate 3D Model", variant="primary", size="lg")

        with gr.Column(scale=1):
            # glb file
            output_model = gr.Model3D(
                label="🎭 3D Model Preview", 
                height=512
            )
            gr.Markdown("""
            ### 📌 Controls:
            - 🖱️ **Left Click & Drag**: Rotate model
            - 🖱️ **Right Click & Drag**: Pan view
            - 🖱️ **Scroll Wheel**: Zoom in/out
            - 📥 Click top-right corner to download GLB file
            """)

    with gr.Row():
        gr.Markdown("### 🖼️ Example Images (Click to Try):")
        gr.Examples(
            examples=[
                ["examples/rabbit.png"],
                ["examples/robot.png"],
                ["examples/teapot.png"],
                ["examples/barrel.png"],
                ["examples/cactus.png"],
                ["examples/cyan_car.png"],
                ["examples/pickup.png"],
                ["examples/swivelchair.png"],
                ["examples/warhammer.png"],
            ],
            fn=process_image,
            inputs=[input_image],
            outputs=[seg_image],
            cache_examples=False
        )
    
    gr.Markdown("""
    ---
    ### ⚠️ Important Notes:
    - Generation takes 1-2 minutes, please be patient
    - Best results with clear, prominent subjects
    - Generated models may need further optimization in professional 3D software
    - Each colored section represents an independent 3D part
    
    ### 🤝 Technical Support:
    Powered by NVIDIA PartPacker technology. For issues, please refer to the [official documentation](https://research.nvidia.com/labs/dir/partpacker/)
    """)

    button_gen.click(
        process_image, inputs=[input_image], outputs=[seg_image]
    ).then(
        get_random_seed, inputs=[randomize_seed, seed], outputs=[seed]
    ).then(
        process_3d, inputs=[seg_image, num_steps, cfg_scale, input_grid_res, seed, simplify_mesh, target_num_faces], outputs=[output_model]
    )

block.launch()