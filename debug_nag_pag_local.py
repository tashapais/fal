#!/usr/bin/env python3
"""
Local debugging script for NAG-PAG implementation
"""

import torch
import torch.nn.functional as F
import math
from typing import Optional, List, Dict, Any
import os

# Simplified NAG-PAG processor for debugging
class DebugNAGPAGProcessor:
    def __init__(self, nag_pag_scale: float = 3.0):
        self.nag_pag_scale = nag_pag_scale
        self.hook_count = 0
        
    def hook_attention_forward(self, module, input, output):
        """Debug hook function to see what's happening"""
        self.hook_count += 1
        print(f"\n🔍 Hook #{self.hook_count} called on {getattr(module, '_module_name', 'unknown')}")
        print(f"  📋 Input type: {type(input)}")
        
        if isinstance(input, tuple):
            print(f"  📋 Tuple length: {len(input)}")
            for i, item in enumerate(input):
                if hasattr(item, 'shape'):
                    print(f"    [{i}]: {type(item)} - shape: {item.shape}")
                else:
                    print(f"    [{i}]: {type(item)}")
        elif hasattr(input, 'shape'):
            print(f"  📋 Input shape: {input.shape}")
            
        print(f"  📋 Output type: {type(output)}")
        if hasattr(output, 'shape'):
            print(f"  📋 Output shape: {output.shape}")
        elif isinstance(output, tuple) and len(output) > 0 and hasattr(output[0], 'shape'):
            print(f"  📋 Output tuple, first element shape: {output[0].shape}")
            
        # Check if module has attention components
        has_q = hasattr(module, 'to_q')
        has_k = hasattr(module, 'to_k') 
        has_v = hasattr(module, 'to_v')
        print(f"  🎯 Attention components: to_q={has_q}, to_k={has_k}, to_v={has_v}")
        
        return output

def test_flux_structure():
    """Test FLUX model structure to understand attention modules"""
    print("🔍 Testing FLUX structure locally...")
    
    try:
        from diffusers import FluxPipeline
        from huggingface_hub import login
        
        # Try to use HF token if available
        hf_token = os.getenv("HF_TOKEN")
        if hf_token:
            login(token=hf_token.strip())
            print("✅ Authenticated with HF")
        
        print("📦 Loading FLUX.1-dev pipeline...")
        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            torch_dtype=torch.bfloat16,
            device_map="auto" if torch.cuda.is_available() else "cpu"
        )
        
        print("🔍 Exploring transformer structure...")
        transformer = pipe.transformer
        
        # Find attention modules
        def find_attention_modules(module, path="", depth=0):
            if depth > 4:
                return
                
            indent = "  " * depth
            print(f"{indent}{path}: {type(module).__name__}")
            
            # Check if this is an attention module
            if hasattr(module, 'to_q') and hasattr(module, 'to_k') and hasattr(module, 'to_v'):
                print(f"{indent}  🎯 ATTENTION MODULE FOUND!")
                return path
                
            # Check children
            attention_paths = []
            for name, child in module.named_children():
                child_path = f"{path}.{name}" if path else name
                result = find_attention_modules(child, child_path, depth + 1)
                if result:
                    attention_paths.append(result)
            
            return attention_paths if attention_paths else None
        
        attention_modules = find_attention_modules(transformer)
        print(f"\n🎯 Found attention modules: {attention_modules}")
        
        # Test with a simple hook on a few modules
        processor = DebugNAGPAGProcessor()
        hooks = []
        
        # Try to hook the first few attention modules we find
        if attention_modules:
            for path in attention_modules[:2] if isinstance(attention_modules, list) else [attention_modules]:
                try:
                    # Navigate to module
                    parts = path.split('.')
                    module = transformer
                    for part in parts:
                        if part.isdigit():
                            module = module[int(part)]
                        else:
                            module = getattr(module, part)
                    
                    module._module_name = path
                    hook = module.register_forward_hook(processor.hook_attention_forward)
                    hooks.append(hook)
                    print(f"✅ Hooked {path}")
                    
                except Exception as e:
                    print(f"❌ Failed to hook {path}: {e}")
        
        # Run a simple generation to see what happens
        print("\n🧪 Running test generation...")
        result = pipe(
            "a simple test image",
            height=512,
            width=512,
            num_inference_steps=2,  # Very quick test
            generator=torch.Generator().manual_seed(42)
        )
        
        print(f"\n📊 Hook was called {processor.hook_count} times")
        
        # Cleanup
        for hook in hooks:
            hook.remove()
            
        return True
        
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("💡 Please install: pip install diffusers transformers torch")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_simple_attention_hook():
    """Test a simple attention hook to understand the input structure"""
    print("\n🔍 Testing simple attention hook...")
    
    class SimpleAttention(torch.nn.Module):
        def __init__(self, dim=512):
            super().__init__()
            self.to_q = torch.nn.Linear(dim, dim)
            self.to_k = torch.nn.Linear(dim, dim)
            self.to_v = torch.nn.Linear(dim, dim)
            
        def forward(self, x):
            q = self.to_q(x)
            k = self.to_k(x) 
            v = self.to_v(x)
            return q, k, v
    
    # Create test module
    attn = SimpleAttention()
    processor = DebugNAGPAGProcessor()
    
    # Hook it
    hook = attn.register_forward_hook(processor.hook_attention_forward)
    
    # Test with sample input
    x = torch.randn(2, 100, 512)  # batch, seq_len, dim
    print(f"📊 Input shape: {x.shape}")
    
    with torch.no_grad():
        output = attn(x)
        
    print(f"📊 Output: {type(output)}, shapes: {[o.shape for o in output]}")
    
    hook.remove()

if __name__ == "__main__":
    print("🚀 NAG-PAG Local Debugging")
    print("=" * 50)
    
    # Test simple attention first
    test_simple_attention_hook()
    
    # Test FLUX if available
    if torch.cuda.is_available():
        print(f"\n💻 CUDA available, testing FLUX...")
        test_flux_structure()
    else:
        print(f"\n💻 No CUDA, skipping FLUX test") 