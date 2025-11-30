from diffusers import StableDiffusionPipeline
import torch

class StableDiffusionGenerator:
    def __init__(self, model_name="runwayml/stable-diffusion-v1-5", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.pipe = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float16 if self.device=="cuda" else torch.float32)
        self.pipe = self.pipe.to(self.device)

    def generate_image(self, prompt: str, num_inference_steps=50, guidance_scale=7.5):
        with torch.autocast(self.device):
            image = self.pipe(
                prompt, num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale
            ).images[0]
        return image
