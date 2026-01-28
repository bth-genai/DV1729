import re
import os
import json
from ollama import chat
from ollama import ChatResponse
import torch

# Choose pipeline based on model
#from diffusers import StableDiffusionPipeline, DDPMScheduler
from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler

# Schema for structured response
schema = {
  'type': 'object',
  'properties': {
    'answer': {'type': 'string'},
    'prompt': {'type': 'string'},
    'filename': {
        'type': 'string'
    },
  },
  'required': ['answer', 'prompt', 'filename']
}
  
#Model
model = "llama3.2"

#Function for removing "think section" in LLM-answers (change RE as needed)
def removeThinkSection(text):
    responseString = re.sub(r'<think>.+?</think>', '', text, flags=re.DOTALL)
    return (responseString)

def main():
    print("***This application makes an LLM create a prompt which then a diffuser model generates an image from***\n")
    
    #Get initial question from user
    question = input("Please state your question: ")
    
    response: ChatResponse = chat(model=model, messages=[{
        'role': 'user',
        'content': "Based on this question: '" + question + "'. Give a short critical answer and create a prompt"
            + " for generating an image using stable diffusion. The image should examplify the answer."
            + "Also give the image a short filename without type extension ."
    }],
        format=schema
    )

    data = json.loads(response.message.content)
    print("\n\n[" + model + "]: \n")
    print(f"Answer: {data['answer']}")
    print(f"Prompt: {data['prompt']}")
    print(f"Filename: {data['filename']}")
    
    generate_image(data['prompt'], data['filename'])

    
def generate_image(prompt, filename):
    if not filename.lower().endswith(".png"):
        filename += ".png"

    # Choose pipeline based on model
    #model = "./data/DiffusionModels/realisticVisionV60B1_v51HyperVAE.safetensors"
    #pipe = StableDiffusionPipeline.from_single_file(model, torch_dtype=torch.float16)
    model = "./data/DiffusionModels/cyberrealisticXL_v80.safetensors"
    pipe = StableDiffusionXLPipeline.from_single_file(model, torch_dtype=torch.float16)

    pipe.to("cuda")
    # Choose scheduler based on model
    #scheduler = DDPMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear")
    scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.scheduler = scheduler
    image = pipe(
        prompt,
        num_inference_steps=30,
        guidance_scale=7.5,
    ).images[0]

    file_path =  os.path.join(os.path.dirname(__file__), filename)
    image.save(file_path)


if __name__ == "__main__":
    main()
