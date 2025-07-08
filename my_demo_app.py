import fal
from pydantic import BaseModel, Field
from fal.toolkit import Image


class Input(BaseModel):
    prompt: str = Field(
        description="The prompt to generate an image from",
        examples=["A beautiful image of a cat"],
    )


class Output(BaseModel):
    image: Image


class MyApp(fal.App, keep_alive=300, name="my-demo-app"):
    machine_type = "GPU-H100"
    requirements = [
        "hf-transfer==0.1.9",
        "diffusers[torch]==0.32.2",
        "transformers[sentencepiece]==4.51.0",
        "accelerate==1.6.0",
    ]

    def setup(self):
        # Enable HF Transfer for faster downloads
        import os
        
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        
        import torch
        from diffusers import StableDiffusionXLPipeline
        
        # Load any model you want, we'll use stabilityai/stable-diffusion-xl-base-1.0
        # Huggingface models will be automatically downloaded to
        # the persistent storage of your account
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
        ).to("cuda")
        
        # Warmup the model before the first request
        self.warmup()

    def warmup(self):
        self.pipe("A beautiful image of a cat")

    @fal.endpoint("/")
    def run(self, request: Input) -> Output:
        result = self.pipe(request.prompt)
        image = Image.from_pil(result.images[0])
        return Output(image=image) 