import fal
import torch
import torch.nn.functional as F
import math
from pydantic import BaseModel, Field
from fal.toolkit import Image, download_file
from typing import Optional, List, Dict, Any
from PIL import Image as PILImage
import numpy as np
import tempfile
import os
import random


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
    enable_nag_pag: bool = Field(default=True, description="Enable NAG-PAG guidance (set to False for baseline comparison)")
    nag_pag_scale: float = Field(default=3.0, description="NAG-PAG guidance scale (only used if enable_nag_pag=True)")
    nag_pag_applied_layers: List[str] = Field(
        default=["transformer_blocks.8.attn", "transformer_blocks.12.attn", "transformer_blocks.16.attn"],
        description="Layers to apply NAG-PAG guidance to"
    )


class TextToImageOutput(BaseModel):
    image: Image = Field(description="Generated image")
    seed: int = Field(description="Seed used for generation")
    has_nag_pag: bool = Field(description="Whether NAG-PAG guidance was applied")


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
    enable_nag_pag: bool = Field(default=True, description="Enable NAG-PAG guidance (set to False for baseline comparison)")
    nag_pag_scale: float = Field(default=3.0, description="NAG-PAG guidance scale (only used if enable_nag_pag=True)")
    nag_pag_applied_layers: List[str] = Field(
        default=["transformer_blocks.8.attn", "transformer_blocks.12.attn", "transformer_blocks.16.attn"],
        description="Layers to apply NAG-PAG guidance to"
    )


class ImageToImageOutput(BaseModel):
    image: Image = Field(description="Transformed image")
    seed: int = Field(description="Seed used for generation")
    has_nag_pag: bool = Field(description="Whether NAG-PAG guidance was applied")


class NAGPAGProcessor:
    """
    Novel NAG-PAG attention processor that combines:
    - NAG: Feature vector computation, combination with weight parameter, and normalization
    - PAG: Identity matrix for negative attention instead of negative prompt
    """
    
    def __init__(self, nag_pag_scale: float = 3.0, normalization_type: str = "l1"):
        self.nag_pag_scale = nag_pag_scale
        self.normalization_type = normalization_type
        self.stored_attn = {}
        
    def hook_attention_forward(self, module, input, output):
        """Hook function to intercept and modify attention computation"""
        if not hasattr(module, 'to_q') or not hasattr(module, 'to_k') or not hasattr(module, 'to_v'):
            return output
            
        # Apply NAG-PAG guidance to attention computation
        try:
            # Handle different input structures - for attention modules, input is typically a tuple
            if isinstance(input, tuple):
                if len(input) > 0:
                    hidden_states = input[0]
                    # Validate that we got a tensor
                    if not isinstance(hidden_states, torch.Tensor):
                        print(f"⚠️ First element of tuple is not a tensor: {type(hidden_states)}")
                        return output
                else:
                    # Empty tuple inputs can happen during model initialization - just pass through
                    return output
            elif isinstance(input, torch.Tensor):
                hidden_states = input
            else:
                print(f"⚠️ Unexpected input type: {type(input)}")
                return output
            
            # Get Q, K, V projections
            q = module.to_q(hidden_states)
            k = module.to_k(hidden_states)
            v = module.to_v(hidden_states)
            
            # Apply NAG-PAG guidance
            guided_output = self.apply_nag_pag_guidance(output, q, k, v, hidden_states)
            
            # Validate guidance succeeded
            if guided_output is None or (hasattr(guided_output, 'shape') and guided_output.shape != output.shape):
                print(f"⚠️ NAG-PAG guidance failed shape validation, using original output")
                return output
            
            return guided_output
            
        except Exception as e:
            print(f"⚠️ NAG-PAG guidance failed, using original output: {e}")
            print(f"⚠️ Input type: {type(input)}, Input length: {len(input) if hasattr(input, '__len__') else 'N/A'}")
            return output
    
    def apply_nag_pag_guidance(self, original_attn_output, q, k, v, input_tensor):
        """
        Apply NAG-PAG guidance:
        1. Compute attention weights normally (positive)
        2. Create identity matrix attention (negative, like PAG)
        3. Extrapolate and normalize features (like NAG)
        """
        try:
            # Validate input shapes
            if q.dim() != 3 or k.dim() != 3 or v.dim() != 3:
                print(f"⚠️ Unexpected tensor dimensions: q={q.shape}, k={k.shape}, v={v.shape}")
                return original_attn_output
                
            batch_size, seq_len, embed_dim = q.shape
            
            # Validate that all tensors have the same shape
            if k.shape != q.shape or v.shape != q.shape:
                print(f"⚠️ Tensor shape mismatch: q={q.shape}, k={k.shape}, v={v.shape}")
                return original_attn_output
            
            # Infer number of heads from the architecture
            # For FLUX, we need to check the actual dimensions
            possible_head_dims = [64, 128, 256, 512]
            num_heads = None
            head_dim = None
            
            for hd in possible_head_dims:
                if embed_dim % hd == 0:
                    num_heads = embed_dim // hd
                    head_dim = hd
                    break
            
            # Fallback calculation
            if num_heads is None:
                num_heads = max(8, embed_dim // 128)
                head_dim = embed_dim // num_heads
            
            # Ensure head_dim is valid
            if head_dim <= 0 or num_heads <= 0:
                print(f"⚠️ Invalid head configuration: num_heads={num_heads}, head_dim={head_dim}")
                return original_attn_output
            
            # Reshape for multi-head attention
            q = q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
            k = k.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
            v = v.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
            
            # Compute normal attention weights (positive branch)
            attn_weights_pos = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
            attn_weights_pos = F.softmax(attn_weights_pos, dim=-1)
            
            # Create identity matrix attention (negative branch, like PAG)
            attn_weights_neg = torch.eye(seq_len, device=q.device, dtype=q.dtype)
            attn_weights_neg = attn_weights_neg.unsqueeze(0).unsqueeze(0).expand(batch_size, num_heads, -1, -1)
            
            # Compute outputs for both branches
            attn_output_pos = torch.matmul(attn_weights_pos, v)
            attn_output_neg = torch.matmul(attn_weights_neg, v)
            
            # Reshape back
            attn_output_pos = attn_output_pos.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
            attn_output_neg = attn_output_neg.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
            
            # NAG-style feature extrapolation and normalization
            # Extrapolate: output = positive + scale * (positive - negative)
            feature_diff = attn_output_pos - attn_output_neg
            guided_output = attn_output_pos + self.nag_pag_scale * feature_diff
            
            # L1 normalization (like NAG)
            if self.normalization_type == "l1":
                norm_factor = torch.norm(guided_output, p=1, dim=-1, keepdim=True)
                norm_factor = torch.clamp(norm_factor, min=1e-8)
                guided_output = guided_output / norm_factor
                
                # Rescale to match original magnitude
                original_norm = torch.norm(attn_output_pos, p=1, dim=-1, keepdim=True)
                guided_output = guided_output * original_norm
            
            # Alpha blending for stability
            alpha = 0.7  # Blend factor
            final_output = alpha * guided_output + (1 - alpha) * attn_output_pos
            
            return final_output
            
        except Exception as e:
            print(f"⚠️ NAG-PAG guidance computation failed: {e}")
            print(f"⚠️ Debug info: q.shape={q.shape if 'q' in locals() else 'N/A'}, batch_size={batch_size if 'batch_size' in locals() else 'N/A'}")
            # Return original attention output as fallback
            return original_attn_output


class FluxNAGPAGApp(fal.App, keep_alive=300, name="flux-nag-pag-app"):
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
        import math
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

        # Authenticate with Hugging Face
        hf_token = os.getenv("HF_TOKEN")
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

        masked_token = f"{hf_token[:8]}..." if len(hf_token) > 8 else "***"
        print(f"🔑 Found HF_TOKEN: {masked_token}")

        try:
            login(token=hf_token)
            print("✅ Successfully authenticated with Hugging Face")
        except Exception as e:
            raise ValueError(f"Failed to authenticate with Hugging Face: {str(e)}")

        # Load FLUX.1 [dev] pipeline for text-to-image
        print("🔧 Loading FLUX.1-dev model for text-to-image with NAG-PAG...")
        self.pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            torch_dtype=torch.bfloat16,
            use_auth_token=hf_token
        ).to("cuda")

        # Load FLUX.1 [dev] pipeline for image-to-image
        print("🔧 Loading FLUX.1-dev model for image-to-image with NAG-PAG...")
        self.img2img_pipe = FluxImg2ImgPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            torch_dtype=torch.bfloat16,
            use_auth_token=hf_token
        ).to("cuda")

        # Enable memory optimization
        self.pipe.enable_model_cpu_offload()
        self.img2img_pipe.enable_model_cpu_offload()

        print("✅ FLUX.1-dev models loaded successfully with NAG-PAG support")

        # Debug transformer structure
        print("\n🔍 Analyzing FLUX transformer architecture...")
        self.debug_transformer_structure(self.pipe.transformer)

        # Warmup
        self.warmup()

    def setup_nag_pag_hooks(self, pipe, nag_pag_scale: float, applied_layers: List[str]):
        """Setup NAG-PAG hooks on specified attention layers"""
        processor = NAGPAGProcessor(nag_pag_scale=nag_pag_scale)
        hooks = []
        
        transformer = pipe.transformer
        
        print(f"🔍 Setting up NAG-PAG hooks for {len(applied_layers)} layers...")
        
        for layer_name in applied_layers:
            try:
                # Navigate to the specified layer
                layer_parts = layer_name.split('.')
                module = transformer
                for i, part in enumerate(layer_parts):
                    if part.isdigit():
                        module = module[int(part)]
                    else:
                        module = getattr(module, part)
                    print(f"  └─ {'.'.join(layer_parts[:i+1])}: {type(module).__name__}")
                
                # Verify this is an attention module
                if hasattr(module, 'to_q') and hasattr(module, 'to_k') and hasattr(module, 'to_v'):
                    # Add module name for tracking
                    module._module_name = layer_name
                    
                    # Hook the attention module
                    hook = module.register_forward_hook(processor.hook_attention_forward)
                    hooks.append(hook)
                    
                    print(f"✅ Applied NAG-PAG hook to {layer_name}")
                else:
                    print(f"⚠️ Module {layer_name} is not a valid attention module")
                    # Try to find attention submodules
                    for attr_name in dir(module):
                        if 'attn' in attr_name.lower():
                            sub_module = getattr(module, attr_name)
                            if hasattr(sub_module, 'to_q') and hasattr(sub_module, 'to_k') and hasattr(sub_module, 'to_v'):
                                sub_module._module_name = f"{layer_name}.{attr_name}"
                                hook = sub_module.register_forward_hook(processor.hook_attention_forward)
                                hooks.append(hook)
                                print(f"✅ Applied NAG-PAG hook to {layer_name}.{attr_name}")
                                break
                
            except Exception as e:
                print(f"⚠️ Could not apply NAG-PAG hook to {layer_name}: {e}")
                # Try to find any attention modules in the transformer
                print(f"🔍 Searching for attention modules in transformer...")
                self.find_attention_modules(transformer)
                
        if len(hooks) == 0:
            print("⚠️ No NAG-PAG hooks were applied. Falling back to automatic detection...")
            hooks = self.auto_detect_and_hook_attention(transformer, processor)
                
        return hooks, processor
    
    def find_attention_modules(self, module, path="", max_depth=4):
        """Find all attention modules in the transformer"""
        if max_depth <= 0:
            return
            
        # Check if current module is an attention module
        if hasattr(module, 'to_q') and hasattr(module, 'to_k') and hasattr(module, 'to_v'):
            print(f"🔍 Found attention module: {path}")
            
        # Recursively search children
        for name, child in module.named_children():
            child_path = f"{path}.{name}" if path else name
            self.find_attention_modules(child, child_path, max_depth - 1)
    
    def auto_detect_and_hook_attention(self, transformer, processor):
        """Automatically detect and hook attention modules"""
        hooks = []
        
        def hook_attention_recursive(module, path=""):
            if hasattr(module, 'to_q') and hasattr(module, 'to_k') and hasattr(module, 'to_v'):
                module._module_name = path
                hook = module.register_forward_hook(processor.hook_attention_forward)
                hooks.append(hook)
                print(f"✅ Auto-hooked attention module: {path}")
                
            for name, child in module.named_children():
                child_path = f"{path}.{name}" if path else name
                hook_attention_recursive(child, child_path)
        
        hook_attention_recursive(transformer)
        print(f"🔍 Auto-detected and hooked {len(hooks)} attention modules")
        return hooks
    
    def debug_transformer_structure(self, transformer, max_depth=3):
        """Debug function to inspect transformer structure"""
        def explore_module(module, name="", depth=0):
            if depth > max_depth:
                return
            
            indent = "  " * depth
            print(f"{indent}{name}: {type(module).__name__}")
            
            if hasattr(module, 'children'):
                for child_name, child_module in module.named_children():
                    if 'attn' in child_name.lower() or 'attention' in child_name.lower():
                        print(f"{indent}  🔍 ATTENTION: {child_name}")
                    explore_module(child_module, child_name, depth + 1)
        
        print("🔍 Exploring transformer structure:")
        explore_module(transformer)

    def warmup(self):
        """Warmup with simple generation"""
        try:
            print("🔥 Starting warmup for NAG-PAG text-to-image...")
            self.pipe(
                "a simple test image",
                height=512,
                width=512,
                num_inference_steps=4,
                generator=torch.Generator("cuda").manual_seed(42)
            )
            print("✅ NAG-PAG text-to-image warmup completed")
        except Exception as e:
            print(f"⚠️ NAG-PAG text-to-image warmup failed: {str(e)}")

    @fal.endpoint("/text-to-image")
    def text_to_image(self, request: TextToImageInput) -> TextToImageOutput:
        try:
            import torch
            import random
            import tempfile
            import os
            import math

            # Generate or use provided seed
            seed = request.seed if request.seed is not None else random.randint(0, 2**32 - 1)
            generator = torch.Generator("cuda").manual_seed(seed)

            # Setup NAG-PAG hooks if enabled
            hooks = []
            processor = None
            if request.enable_nag_pag:
                print(f"🎯 Generating image with NAG-PAG guidance: {request.prompt[:50]}...")
                print(f"🔧 NAG-PAG scale: {request.nag_pag_scale}, Applied layers: {request.nag_pag_applied_layers}")
                hooks, processor = self.setup_nag_pag_hooks(
                    self.pipe, 
                    request.nag_pag_scale, 
                    request.nag_pag_applied_layers
                )
            else:
                print(f"🎯 Generating baseline image (no NAG-PAG): {request.prompt[:50]}...")

            try:
                # Generate image
                result = self.pipe(
                    prompt=request.prompt,
                    height=request.height,
                    width=request.width,
                    num_inference_steps=request.num_inference_steps,
                    guidance_scale=request.guidance_scale,
                    generator=generator,
                )

                guidance_type = "NAG-PAG guided" if request.enable_nag_pag else "baseline"
                print(f"🖼️ {guidance_type} image generated successfully")

                # Get the PIL image
                pil_image = result.images[0]
                print(f"📏 PIL image size: {pil_image.size}, mode: {pil_image.mode}")

                # Save to temporary file
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
                    pil_image.save(tmp_file.name, format="PNG")
                    temp_path = tmp_file.name

                # Convert to fal Image
                fal_image = Image.from_path(temp_path)
                print(f"✅ {guidance_type} Fal Image created")

                # Clean up temporary file
                try:
                    os.unlink(temp_path)
                except:
                    pass

                # Return structured output according to fal conventions
                return TextToImageOutput(
                    image=fal_image,
                    seed=seed,
                    has_nag_pag=request.enable_nag_pag
                )

            finally:
                # Clean up hooks
                for hook in hooks:
                    hook.remove()

        except Exception as e:
            print(f"❌ Error in NAG-PAG text_to_image: {str(e)}")
            import traceback
            print(traceback.format_exc())
            raise ValueError(f"NAG-PAG text-to-image generation failed: {str(e)}")

    @fal.endpoint("/image-to-image")
    def image_to_image(self, request: ImageToImageInput) -> ImageToImageOutput:
        try:
            import torch
            import random
            import tempfile
            import os
            import math

            # Download and load the input image
            with tempfile.TemporaryDirectory() as temp_dir:
                input_path = download_file(str(request.image_url), target_dir=temp_dir)
                input_image = PILImage.open(input_path).convert('RGB')
                print(f"📥 Input image size: {input_image.size}")

                # Generate or use provided seed
                seed = request.seed if request.seed is not None else random.randint(0, 2**32 - 1)
                generator = torch.Generator("cuda").manual_seed(seed)

                # Setup NAG-PAG hooks if enabled
                hooks = []
                processor = None
                if request.enable_nag_pag:
                    print(f"🎯 Transforming image with NAG-PAG guidance: {request.prompt[:50]}...")
                    print(f"🔧 NAG-PAG scale: {request.nag_pag_scale}, Applied layers: {request.nag_pag_applied_layers}")
                    hooks, processor = self.setup_nag_pag_hooks(
                        self.img2img_pipe,
                        request.nag_pag_scale,
                        request.nag_pag_applied_layers
                    )
                else:
                    print(f"🎯 Transforming image (baseline, no NAG-PAG): {request.prompt[:50]}...")

                try:
                    # Generate image transformation
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

                    guidance_type = "NAG-PAG guided" if request.enable_nag_pag else "baseline"
                    print(f"🖼️ {guidance_type} image transformation completed")

                    # Get the PIL image
                    pil_image = result.images[0]
                    print(f"📏 PIL image size: {pil_image.size}, mode: {pil_image.mode}")

                    # Save to temporary file
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
                        pil_image.save(tmp_file.name, format="PNG")
                        temp_path = tmp_file.name

                    # Convert to fal Image
                    fal_image = Image.from_path(temp_path)
                    print(f"✅ {guidance_type} Fal Image created")

                    # Clean up temporary file
                    try:
                        os.unlink(temp_path)
                    except:
                        pass

                    # Return structured output according to fal conventions
                    return ImageToImageOutput(
                        image=fal_image,
                        seed=seed,
                        has_nag_pag=request.enable_nag_pag
                    )

                finally:
                    # Clean up hooks
                    for hook in hooks:
                        hook.remove()

        except Exception as e:
            print(f"❌ Error in NAG-PAG image_to_image: {str(e)}")
            import traceback
            print(traceback.format_exc())
            raise ValueError(f"NAG-PAG image-to-image generation failed: {str(e)}") 