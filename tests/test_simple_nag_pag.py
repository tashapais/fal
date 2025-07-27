#!/usr/bin/env python3
"""
Simple authenticated test for NAG-PAG FLUX implementation
"""

import fal_client
import os

def test_nag_pag_authenticated():
    """Test NAG-PAG with proper fal authentication"""
    
    print("🧪 Testing NAG-PAG Text-to-Image with Authentication...")
    
    try:
        # Test text-to-image with NAG-PAG
        result = fal_client.run(
            "tashapais/flux-nag-pag-app/text-to-image",
            arguments={
                "prompt": "A serene mountain landscape with a crystal clear lake, golden hour lighting",
                "width": 512,
                "height": 512,
                "num_inference_steps": 10,  # Quick test
                "guidance_scale": 3.5,
                "nag_pag_scale": 3.0,  # NAG-PAG guidance strength
                "nag_pag_applied_layers": [
                    "transformer_blocks.8.attn", 
                    "transformer_blocks.12.attn"
                ],
                "seed": 42
            }
        )
        
        print("✅ NAG-PAG Text-to-Image successful!")
        print(f"🔗 Image URL: {result.url}")
        
        # Download the image
        import requests
        from PIL import Image
        import io
        
        response = requests.get(result.url)
        if response.status_code == 200:
            image = Image.open(io.BytesIO(response.content))
            image.save("nag_pag_test_output.png")
            print("💾 Image saved as 'nag_pag_test_output.png'")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_comparison():
    """Test comparison between normal FLUX and NAG-PAG"""
    
    print("\n🔬 Comparing Normal FLUX vs NAG-PAG...")
    
    prompt = "A beautiful cat sitting in a garden"
    
    try:
        # Test with NAG-PAG (high guidance)
        print("  🎯 Testing with NAG-PAG guidance...")
        nag_pag_result = fal_client.run(
            "tashapais/flux-nag-pag-app/text-to-image",
            arguments={
                "prompt": prompt,
                "width": 512,
                "height": 512,
                "num_inference_steps": 8,
                "guidance_scale": 3.5,
                "nag_pag_scale": 5.0,  # Strong NAG-PAG guidance
                "nag_pag_applied_layers": ["transformer_blocks.12.attn"],
                "seed": 123
            }
        )
        
        # Test with minimal NAG-PAG (essentially normal FLUX)
        print("  🎯 Testing with minimal NAG-PAG (baseline)...")
        baseline_result = fal_client.run(
            "tashapais/flux-nag-pag-app/text-to-image",
            arguments={
                "prompt": prompt,
                "width": 512,
                "height": 512,
                "num_inference_steps": 8,
                "guidance_scale": 3.5,
                "nag_pag_scale": 0.0,  # No NAG-PAG guidance
                "nag_pag_applied_layers": [],  # No layers
                "seed": 123  # Same seed for comparison
            }
        )
        
        print("✅ Comparison test successful!")
        print(f"🔗 NAG-PAG result: {nag_pag_result.url}")
        print(f"🔗 Baseline result: {baseline_result.url}")
        
        # Download both for comparison
        import requests
        from PIL import Image
        import io
        
        for name, result in [("nag_pag", nag_pag_result), ("baseline", baseline_result)]:
            response = requests.get(result.url)
            if response.status_code == 200:
                image = Image.open(io.BytesIO(response.content))
                image.save(f"comparison_{name}.png")
                print(f"💾 {name.title()} image saved as 'comparison_{name}.png'")
        
        return True
        
    except Exception as e:
        print(f"❌ Comparison error: {e}")
        return False

if __name__ == "__main__":
    print("🚀 NAG-PAG Authentication Test")
    print("=" * 50)
    
    # Check if FAL_KEY is set
    if not os.getenv("FAL_KEY"):
        print("⚠️  You need to set your FAL_KEY environment variable")
        print("   1. Get your API key from: https://fal.ai/dashboard/keys")
        print("   2. Set it: export FAL_KEY=your_api_key_here")
        print("   3. Or create a .env file with: FAL_KEY=your_api_key_here")
        exit(1)
    
    # Run tests
    success1 = test_nag_pag_authenticated()
    success2 = test_comparison()
    
    print("\n" + "=" * 50)
    if success1 and success2:
        print("🎉 All NAG-PAG tests passed!")
        print("📁 Check the generated images to see the NAG-PAG effect")
    else:
        print("❌ Some tests failed - check the errors above") 