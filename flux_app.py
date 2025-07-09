import fal
from pydantic import BaseModel, Field
from fal.toolkit import Image
from typing import Optional, List
import time


class TextToImageInput(BaseModel):
    prompt: str = Field(
        description="The prompt to generate an image from",
        examples=["A beautiful landscape with mountains and a lake"]
    )
    width: int = Field(default=1024, description="Width of the generated image")
    height: int = Field(default=1024, description="Height of the generated image")
    num_inference_steps: int = Field(default=20, description="Number of inference steps")
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
    num_inference_steps: int = Field(default=20, description="Number of inference steps")
    guidance_scale: float = Field(default=3.5, description="Guidance scale for generation")
    seed: Optional[int] = Field(default=None, description="Random seed for reproducibility")


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
        "huggingface_hub>=0.25.0",
        "python-dotenv>=1.0.0",
    ]

    def setup(self):
        import torch
        import os
        from diffusers import FluxPipeline
        from huggingface_hub import login
        
        # Load .env file if it exists (for local development)
        try:
            from dotenv import load_dotenv
            load_dotenv()
            print("📁 Loaded .env file for local development")
        except ImportError:
            print("💡 python-dotenv not available, using environment variables only")
        except Exception as e:
            print(f"⚠️ Could not load .env file: {e}")
        
        # Authenticate with Hugging Face using token from fal secrets or .env
        hf_token = os.getenv("HF_TOKEN")
        
        # Clean up the token (remove whitespace)
        if hf_token:
            hf_token = hf_token.strip()
        
        if not hf_token:
            raise ValueError(
                "HF_TOKEN not found in environment variables. Please set it using one of these methods:\n"
                "1. For local development: Add HF_TOKEN=your_token_here to your .env file\n"
                "2. For production: fal secrets set HF_TOKEN your_token_here\n"
                "3. Export directly: export HF_TOKEN=your_token_here\n"
                "Get your token from: https://huggingface.co/settings/tokens"
            )
        
        # Mask token for logging (show only first 8 chars)
        masked_token = f"{hf_token[:8]}..." if len(hf_token) > 8 else "***"
        print(f"🔑 Found HF_TOKEN: {masked_token}")
        
        try:
            login(token=hf_token)
            print("✅ Successfully authenticated with Hugging Face")
        except Exception as e:
            raise ValueError(f"Failed to authenticate with Hugging Face: {str(e)}")
        
        # Load FLUX.1 [dev] pipeline
        print("Loading FLUX.1-dev model...")
        self.pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            torch_dtype=torch.bfloat16,
            use_auth_token=hf_token  # Explicitly pass token for extra security
        ).to("cuda")
        
        # Enable memory optimization
        self.pipe.enable_model_cpu_offload()
        
        print("✅ FLUX.1-dev model loaded successfully")
        
        # Warmup with a simple generation
        self.warmup()

    def warmup(self):
        import torch
        # Simple warmup generation for FLUX.1-dev
        try:
            print("🔥 Starting warmup...")
            self.pipe(
                "a simple test image",
                height=512,
                width=512,
                num_inference_steps=4,  # Quick warmup with fewer steps
                generator=torch.Generator("cuda").manual_seed(42)
            )
            print("✅ Warmup completed successfully")
        except Exception as e:
            print(f"⚠️ Warmup failed: {str(e)}")
            # Don't fail the setup if warmup fails
            pass

    @fal.endpoint("/text-to-image")
    def text_to_image(self, request: TextToImageInput) -> Image:
        try:
            import torch
            import random
            import io
            
            print(f"🎯 Generating image with prompt: {request.prompt[:50]}...")
            
            # Generate or use provided seed
            seed = request.seed if request.seed is not None else random.randint(0, 2**32 - 1)
            generator = torch.Generator("cuda").manual_seed(seed)
            
            # Generate image
            result = self.pipe(
                prompt=request.prompt,
                height=request.height,
                width=request.width,
                num_inference_steps=request.num_inference_steps,
                guidance_scale=request.guidance_scale,
                generator=generator,
            )
            
            print("🖼️ Image generated successfully, converting to fal Image...")
            
            # Get the PIL image
            pil_image = result.images[0]
            print(f"📏 PIL image size: {pil_image.size}, mode: {pil_image.mode}")
            
            # Convert to fal Image
            fal_image = Image.from_pil(pil_image)
            print(f"✅ Fal Image created, URL: {getattr(fal_image, 'url', 'NO URL ATTRIBUTE')}")
            
            # Check if URL is properly set
            if hasattr(fal_image, 'url') and fal_image.url:
                print(f"🔗 Image URL: {fal_image.url}")
            else:
                print("⚠️ Warning: fal Image has no URL or empty URL!")
                # Try to debug what attributes the image has
                print(f"📋 Image attributes: {[attr for attr in dir(fal_image) if not attr.startswith('_')]}")
            
            return fal_image
            
        except Exception as e:
            print(f"❌ Error in text_to_image: {str(e)}")
            import traceback
            print(traceback.format_exc())
            raise ValueError(f"Text-to-image generation failed: {str(e)}")

    @fal.endpoint("/image-to-image")
    def image_to_image(self, request: ImageToImageInput) -> Image:
        try:
            import torch
            import random
            from diffusers import FluxImg2ImgPipeline
            from PIL import Image as PILImage
            
            print(f"🎯 Transforming image with prompt: {request.prompt[:50]}...")
            
            # Convert fal Image to PIL Image
            input_image = request.image.to_pil()
            print(f"📥 Input image size: {input_image.size}")
            
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
            
            # Generate or use provided seed
            seed = request.seed if request.seed is not None else random.randint(0, 2**32 - 1)
            generator = torch.Generator("cuda").manual_seed(seed)
            
            # Generate image
            result = img2img_pipe(
                prompt=request.prompt,
                image=input_image,
                strength=request.strength,
                num_inference_steps=request.num_inference_steps,
                guidance_scale=request.guidance_scale,
                generator=generator,
            )
            
            print("🖼️ Image transformation completed, converting to fal Image...")
            
            # Get the PIL image
            pil_image = result.images[0]
            print(f"📏 PIL image size: {pil_image.size}, mode: {pil_image.mode}")
            
            # Convert to fal Image
            fal_image = Image.from_pil(pil_image)
            print(f"✅ Fal Image created, URL: {getattr(fal_image, 'url', 'NO URL ATTRIBUTE')}")
            
            return fal_image
            
        except Exception as e:
            print(f"❌ Error in image_to_image: {str(e)}")
            import traceback
            print(traceback.format_exc())
            raise ValueError(f"Image-to-image generation failed: {str(e)}") 