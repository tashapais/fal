import fal
from pydantic import BaseModel, Field
from fal.toolkit import Image, download_file
from typing import Optional
from PIL import Image as PILImage


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
    image_url: str = Field(description="Input image URL to transform")
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
        "diffusers==0.34.0",
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
        from diffusers import FluxPipeline, FluxImg2ImgPipeline
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

        # Load FLUX.1 [dev] pipeline for text-to-image
        print("Loading FLUX.1-dev model for text-to-image...")
        self.pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            torch_dtype=torch.bfloat16,
            use_auth_token=hf_token  # Explicitly pass token for extra security
        ).to("cuda")

        # Load FLUX.1 [dev] pipeline for image-to-image
        print("Loading FLUX.1-dev model for image-to-image...")
        self.img2img_pipe = FluxImg2ImgPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            torch_dtype=torch.bfloat16,
            use_auth_token=hf_token  # Explicitly pass token for extra security
        ).to("cuda")

        # Enable memory optimization
        self.pipe.enable_model_cpu_offload()
        self.img2img_pipe.enable_model_cpu_offload()

        print("✅ FLUX.1-dev models loaded successfully")

        # Warmup with a simple generation
        self.warmup()

    def warmup(self):
        import torch
        # Simple warmup generation for FLUX.1-dev
        try:
            print("🔥 Starting warmup for text-to-image...")
            self.pipe(
                "a simple test image",
                height=512,
                width=512,
                num_inference_steps=4,  # Quick warmup with fewer steps
                generator=torch.Generator("cuda").manual_seed(42)
            )
            print("✅ Text-to-image warmup completed successfully")
        except Exception as e:
            print(f"⚠️ Text-to-image warmup failed: {str(e)}")
        
        try:
            print("🔥 Starting warmup for image-to-image...")
            # Create a simple test image for warmup
            from PIL import Image as PILImage
            test_image = PILImage.new('RGB', (512, 512), color='white')
            self.img2img_pipe(
                "a simple test image",
                image=test_image,
                strength=0.5,
                height=512,
                width=512,
                num_inference_steps=4,  # Quick warmup with fewer steps
                generator=torch.Generator("cuda").manual_seed(42)
            )
            print("✅ Image-to-image warmup completed successfully")
        except Exception as e:
            print(f"⚠️ Image-to-image warmup failed: {str(e)}")
            # Don't fail the setup if warmup fails
            pass

    @fal.endpoint("/text-to-image")
    def text_to_image(self, request: TextToImageInput) -> Image:
        try:
            import torch
            import random
            import tempfile
            import os

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

            print("🖼️ Image generated successfully, saving and converting to fal Image...")

            # Get the PIL image
            pil_image = result.images[0]
            print(f"📏 PIL image size: {pil_image.size}, mode: {pil_image.mode}")

            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
                pil_image.save(tmp_file.name, format="PNG")
                temp_path = tmp_file.name

            # Convert to fal Image using the file path
            fal_image = Image.from_path(temp_path)
            print(f"✅ Fal Image created, URL: {getattr(fal_image, 'url', 'NO URL ATTRIBUTE')}")

            # Clean up temporary file
            try:
                os.unlink(temp_path)
            except:
                pass

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
            import tempfile
            import os
            
            print(f"🎯 Transforming image with prompt: {request.prompt[:50]}...")

            # Download and load the input image
            with tempfile.TemporaryDirectory() as temp_dir:
                input_path = download_file(str(request.image_url), target_dir=temp_dir)
                input_image = PILImage.open(input_path).convert('RGB')
                print(f"📥 Input image size: {input_image.size}")
                
                # Generate or use provided seed
                seed = request.seed if request.seed is not None else random.randint(0, 2**32 - 1)
                generator = torch.Generator("cuda").manual_seed(seed)
                
                print("🔄 Using proper image-to-image pipeline...")
                
                # Generate image using proper image-to-image pipeline
                result = self.img2img_pipe(
                    prompt=request.prompt,
                    image=input_image,
                    strength=request.strength,
                    height=request.height,
                    width=request.width,
                    num_inference_steps=request.num_inference_steps,
                    guidance_scale=request.guidance_scale,
                    generator=generator,
                )
                
                print("🖼️ Image transformation completed, saving and converting to fal Image...")
                
                # Get the PIL image
                pil_image = result.images[0]
                print(f"📏 PIL image size: {pil_image.size}, mode: {pil_image.mode}")
                
                # Save to temporary file
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
                    pil_image.save(tmp_file.name, format="PNG")
                    temp_path = tmp_file.name
                
                # Convert to fal Image using the file path
                fal_image = Image.from_path(temp_path)
                print(f"✅ Fal Image created, URL: {getattr(fal_image, 'url', 'NO URL ATTRIBUTE')}")
                
                # Clean up temporary file
                try:
                    os.unlink(temp_path)
                except:
                    pass
                
                return fal_image

        except Exception as e:
            print(f"❌ Error in image_to_image: {str(e)}")
            import traceback
            print(traceback.format_exc())
            raise ValueError(f"Image-to-image generation failed: {str(e)}") 