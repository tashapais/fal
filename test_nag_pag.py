#!/usr/bin/env python3
"""
Test script for NAG-PAG FLUX.1 [dev] implementation
"""

import os
import sys
import requests
import json
from PIL import Image
import io

# Test configuration
TEST_PROMPT = "A serene mountain landscape with a crystal clear lake, golden hour lighting"
TEST_IMAGE_URL = "https://picsum.photos/512/512"  # Random test image
BASE_URL = "http://localhost:8000"  # Adjust based on your deployment

def test_text_to_image():
    """Test text-to-image generation with NAG-PAG"""
    print("🧪 Testing NAG-PAG Text-to-Image Generation...")
    
    payload = {
        "prompt": TEST_PROMPT,
        "width": 512,
        "height": 512,
        "num_inference_steps": 10,  # Reduced for testing
        "guidance_scale": 3.5,
        "nag_pag_scale": 3.0,
        "nag_pag_applied_layers": [
            "transformer_blocks.8.attn",
            "transformer_blocks.12.attn"
        ],
        "seed": 42
    }
    
    try:
        response = requests.post(f"{BASE_URL}/text-to-image", json=payload, timeout=120)
        
        if response.status_code == 200:
            print("✅ Text-to-image generation successful!")
            
            # Save the result
            if response.headers.get('content-type') == 'application/json':
                result = response.json()
                print(f"📄 Response: {result}")
            else:
                # If it's an image
                image = Image.open(io.BytesIO(response.content))
                image.save("test_output_t2i.png")
                print("💾 Image saved as test_output_t2i.png")
                
        else:
            print(f"❌ Text-to-image failed with status {response.status_code}")
            print(f"📄 Response: {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Network error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

def test_image_to_image():
    """Test image-to-image transformation with NAG-PAG"""
    print("\n🧪 Testing NAG-PAG Image-to-Image Transformation...")
    
    payload = {
        "prompt": "Transform this into a beautiful watercolor painting",
        "image_url": TEST_IMAGE_URL,
        "strength": 0.7,
        "width": 512,
        "height": 512,
        "num_inference_steps": 10,  # Reduced for testing
        "guidance_scale": 3.5,
        "nag_pag_scale": 3.0,
        "nag_pag_applied_layers": [
            "transformer_blocks.8.attn",
            "transformer_blocks.12.attn"
        ],
        "seed": 42
    }
    
    try:
        response = requests.post(f"{BASE_URL}/image-to-image", json=payload, timeout=120)
        
        if response.status_code == 200:
            print("✅ Image-to-image transformation successful!")
            
            # Save the result
            if response.headers.get('content-type') == 'application/json':
                result = response.json()
                print(f"📄 Response: {result}")
            else:
                # If it's an image
                image = Image.open(io.BytesIO(response.content))
                image.save("test_output_i2i.png")
                print("💾 Image saved as test_output_i2i.png")
                
        else:
            print(f"❌ Image-to-image failed with status {response.status_code}")
            print(f"📄 Response: {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Network error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

def test_parameter_variations():
    """Test different NAG-PAG parameter combinations"""
    print("\n🧪 Testing NAG-PAG Parameter Variations...")
    
    test_cases = [
        {"nag_pag_scale": 1.0, "description": "Low guidance"},
        {"nag_pag_scale": 3.0, "description": "Medium guidance"},
        {"nag_pag_scale": 5.0, "description": "High guidance"},
    ]
    
    for i, test_case in enumerate(test_cases):
        print(f"\n🔬 Test {i+1}: {test_case['description']}")
        
        payload = {
            "prompt": TEST_PROMPT,
            "width": 512,
            "height": 512,
            "num_inference_steps": 8,
            "guidance_scale": 3.5,
            "nag_pag_scale": test_case["nag_pag_scale"],
            "nag_pag_applied_layers": ["transformer_blocks.12.attn"],
            "seed": 42
        }
        
        try:
            response = requests.post(f"{BASE_URL}/text-to-image", json=payload, timeout=120)
            
            if response.status_code == 200:
                print(f"✅ {test_case['description']} successful!")
                
                # Save with descriptive name
                if response.headers.get('content-type') != 'application/json':
                    image = Image.open(io.BytesIO(response.content))
                    filename = f"test_scale_{test_case['nag_pag_scale']}.png"
                    image.save(filename)
                    print(f"💾 Image saved as {filename}")
                    
            else:
                print(f"❌ {test_case['description']} failed with status {response.status_code}")
                
        except Exception as e:
            print(f"❌ {test_case['description']} error: {e}")

def test_layer_variations():
    """Test different layer combinations"""
    print("\n🧪 Testing NAG-PAG Layer Variations...")
    
    layer_tests = [
        {
            "layers": ["transformer_blocks.4.attn"],
            "description": "Early layer only"
        },
        {
            "layers": ["transformer_blocks.12.attn"],
            "description": "Middle layer only"
        },
        {
            "layers": ["transformer_blocks.16.attn"],
            "description": "Late layer only"
        },
        {
            "layers": ["transformer_blocks.8.attn", "transformer_blocks.12.attn", "transformer_blocks.16.attn"],
            "description": "Multiple layers"
        }
    ]
    
    for i, test_case in enumerate(layer_tests):
        print(f"\n🔬 Layer test {i+1}: {test_case['description']}")
        
        payload = {
            "prompt": TEST_PROMPT,
            "width": 512,
            "height": 512,
            "num_inference_steps": 8,
            "guidance_scale": 3.5,
            "nag_pag_scale": 3.0,
            "nag_pag_applied_layers": test_case["layers"],
            "seed": 42
        }
        
        try:
            response = requests.post(f"{BASE_URL}/text-to-image", json=payload, timeout=120)
            
            if response.status_code == 200:
                print(f"✅ {test_case['description']} successful!")
                
                # Save with descriptive name
                if response.headers.get('content-type') != 'application/json':
                    image = Image.open(io.BytesIO(response.content))
                    filename = f"test_layers_{i+1}.png"
                    image.save(filename)
                    print(f"💾 Image saved as {filename}")
                    
            else:
                print(f"❌ {test_case['description']} failed with status {response.status_code}")
                
        except Exception as e:
            print(f"❌ {test_case['description']} error: {e}")

def main():
    """Run all tests"""
    print("🚀 Starting NAG-PAG FLUX.1 [dev] Tests")
    print("=" * 50)
    
    # Check if server is running
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code != 200:
            print(f"❌ Server health check failed. Is the app running at {BASE_URL}?")
            return
    except requests.exceptions.RequestException:
        print(f"❌ Cannot connect to server at {BASE_URL}. Is the app running?")
        return
    
    print("✅ Server is running!")
    
    # Run tests
    test_text_to_image()
    test_image_to_image()
    test_parameter_variations()
    test_layer_variations()
    
    print("\n" + "=" * 50)
    print("🎯 All tests completed!")
    print("📁 Check output images in current directory")

if __name__ == "__main__":
    main() 