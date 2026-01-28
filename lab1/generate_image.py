import os
import torch
from diffusers import StableDiffusionPipeline, DDPMScheduler, EulerDiscreteScheduler

model = "./data/DiffusionModels/realisticVisionV60B1_v51HyperVAE.safetensors"
pipe = StableDiffusionPipeline.from_single_file(model, torch_dtype=torch.float16)

# Load LoRA weights
#lora_path = "./path/your_lora_file.safetensors"
#pipe.load_lora_weights(lora_path)

pipe.to("cuda")
prompt = "A cat sitting next to a bowl of fresh salmon, with a puzzled expression on its face, as if wondering why it doesn't actually love fish. Color palette: muted blues and greens, with a few splashes of orange from the salmon. Style: whimsical and playful, with a touch of skepticism."
scheduler = DDPMScheduler(beta_start=0.00085, beta_end=0.012,
                          beta_schedule="scaled_linear")
pipe.scheduler = scheduler
image = pipe(
    prompt,
    num_inference_steps=30,
    guidance_scale=7.5,
).images[0]

file_path =  os.path.join(os.path.dirname(__file__), "cat.png")
image.save(file_path)