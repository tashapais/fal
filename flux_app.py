import fal
from pydantic import BaseModel, Field
from fal.toolkit import Image
from typing import Optional


class TextToImageInput(BaseModel):
    prompt: str = Field(
        description="The prompt to generate an image from",
        examples=["A beautiful landscape with mountains and a lake"]
    )
    width: int = Field(default=1024, description="Width of the generated image")
    height: int = Field(default=1024, description="Height of the generated image")
    num_inference_steps: int = Field(default=28, description="Number of inference steps")
    guidance_scale: float = Field(default=3.5, description="Guidance scale for generation")
    seed: Optional[int] = Field(default=None, description="Random seed for reproducibility")


class ImageToImageInput(BaseModel):
    prompt: str = Field(
        description="The prompt to guide the image transformation",
        examples=["Transform this into a painting in the style of Van Gogh"]
    )
    image: Image = Field(description="Input image to transform")
    strength: float = Field(default=0.8, description="Strength of transformation (0.0 to 1.0)")
    width: int = Field(default=1024, description="Width of the generated image")
    height: int = Field(default=1024, description="Height of the generated image") 
    num_inference_steps: int = Field(default=28, description="Number of inference steps")
    guidance_scale: float = Field(default=3.5, description="Guidance scale for generation")
    seed: Optional[int] = Field(default=None, description="Random seed for reproducibility")


class ImageOutput(BaseModel):
    image: Image


class FluxApp(fal.App, keep_alive=300, name="flux-dev-app"):
    machine_type = "GPU-H100"
    requirements = [
        "torch==2.4.0",
        "diffusers==0.30.3",
        "transformers==4.44.2",
        "accelerate==0.34.2",
        "sentencepiece==0.2.0",
        "protobuf==5.27.3",
        "pillow>=10.0.0",
    ]

    def setup(self):
        import torch
        from diffusers import FluxPipeline
        
        # Load FLUX.1 [dev] pipeline
        self.pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            torch_dtype=torch.bfloat16
        ).to("cuda")
        
        # Enable memory efficient attention
        self.pipe.enable_model_cpu_offload()
        
        # Warmup with a simple generation
        self.warmup()

    def warmup(self):
        import torch
        # Simple warmup generation
        self.pipe(
            "a simple test image",
            height=512,
            width=512,
            num_inference_steps=4,
            generator=torch.Generator("cuda").manual_seed(42)
        )

    @fal.endpoint("/text-to-image")
    def text_to_image(self, request: TextToImageInput) -> ImageOutput:
        try:
            import torch
            
            # Set up generator for reproducibility if seed is provided
            generator = None
            if request.seed is not None:
                generator = torch.Generator("cuda").manual_seed(request.seed)
            
            # Generate image
            result = self.pipe(
                prompt=request.prompt,
                height=request.height,
                width=request.width,
                num_inference_steps=request.num_inference_steps,
                guidance_scale=request.guidance_scale,
                generator=generator,
            )
            
            # Convert to fal Image format
            image = Image.from_pil(result.images[0])
            return ImageOutput(image=image)
            
        except Exception as e:
            raise ValueError(f"Text-to-image generation failed: {str(e)}")

    @fal.endpoint("/image-to-image")
    def image_to_image(self, request: ImageToImageInput) -> ImageOutput:
        try:
            import torch
            from diffusers import FluxImg2ImgPipeline
            from PIL import Image as PILImage
            
            # Convert fal Image to PIL Image
            input_image = request.image.to_pil()
            
            # Resize input image to target dimensions
            input_image = input_image.resize((request.width, request.height))
            
            # Create img2img pipeline from existing pipeline
            img2img_pipe = FluxImg2ImgPipeline(
                transformer=self.pipe.transformer,
                scheduler=self.pipe.scheduler,
                vae=self.pipe.vae,
                text_encoder=self.pipe.text_encoder,
                text_encoder_2=self.pipe.text_encoder_2,
                tokenizer=self.pipe.tokenizer,
                tokenizer_2=self.pipe.tokenizer_2,
            ).to("cuda")
            
            # Set up generator for reproducibility if seed is provided
            generator = None
            if request.seed is not None:
                generator = torch.Generator("cuda").manual_seed(request.seed)
            
            # Generate image
            result = img2img_pipe(
                prompt=request.prompt,
                image=input_image,
                strength=request.strength,
                num_inference_steps=request.num_inference_steps,
                guidance_scale=request.guidance_scale,
                generator=generator,
            )
            
            # Convert to fal Image format
            image = Image.from_pil(result.images[0])
            return ImageOutput(image=image)
            
        except Exception as e:
            raise ValueError(f"Image-to-image generation failed: {str(e)}") 