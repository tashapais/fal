# FLUX.1 [dev] with Novel NAG-PAG Attention Guidance

This implementation combines ideas from **Normalized Attention Guidance (NAG)** and **Perturbed Attention Guidance (PAG)** to create a novel attention mechanism for FLUX.1 [dev] diffusion models.

## 🎯 What is NAG-PAG?

Our novel approach combines the best of both techniques:

### From NAG (Normalized Attention Guidance):
- ✅ **Feature vector computation**: Computes attention feature vectors 
- ✅ **Weight parameter combination**: Combines features with learnable weights
- ✅ **L1 normalization**: Normalizes attention outputs for stability

### From PAG (Perturbed Attention Guidance):
- ✅ **Identity matrix for negative attention**: Uses identity matrix instead of negative prompts
- ✅ **Training-free guidance**: No additional training required
- ✅ **Universal applicability**: Works across different model architectures

## 🔧 Key Innovation

Instead of using negative prompts (like traditional CFG), NAG-PAG creates a "negative" attention branch using an **identity matrix**, then extrapolates between the positive and negative branches using NAG-style normalization.

```python
# Traditional approach: Use negative prompts
positive_output = attention(prompt)
negative_output = attention(negative_prompt)  # Requires negative prompts

# NAG-PAG approach: Use identity matrix
positive_output = attention(prompt)
negative_output = attention_with_identity_matrix(prompt)  # No negative prompts needed
```

## 🚀 Usage

### Text-to-Image Generation

```python
import requests

# Configure NAG-PAG parameters
request_data = {
    "prompt": "A beautiful landscape with mountains and a lake",
    "width": 1024,
    "height": 1024,
    "num_inference_steps": 20,
    "guidance_scale": 3.5,
    "nag_pag_scale": 3.0,  # NAG-PAG guidance strength
    "nag_pag_applied_layers": [
        "transformer_blocks.8.attn",
        "transformer_blocks.12.attn", 
        "transformer_blocks.16.attn"
    ],
    "seed": 42
}

response = requests.post("https://your-app-url/text-to-image", json=request_data)
```

### Image-to-Image Transformation

```python
request_data = {
    "prompt": "Transform this into a painting in the style of Van Gogh",
    "image_url": "https://example.com/input-image.jpg",
    "strength": 0.8,
    "width": 1024,
    "height": 1024,
    "num_inference_steps": 20,
    "guidance_scale": 3.5,
    "nag_pag_scale": 3.0,
    "nag_pag_applied_layers": [
        "transformer_blocks.8.attn",
        "transformer_blocks.12.attn",
        "transformer_blocks.16.attn"
    ],
    "seed": 42
}

response = requests.post("https://your-app-url/image-to-image", json=request_data)
```

## ⚙️ Parameters

### NAG-PAG Specific Parameters

- **`nag_pag_scale`** (float, default: 3.0): Controls the strength of NAG-PAG guidance
  - Higher values = stronger guidance effect
  - Lower values = more subtle guidance
  - Range: 0.0 to 10.0

- **`nag_pag_applied_layers`** (List[str]): Specifies which transformer layers to apply guidance to
  - Default: `["transformer_blocks.8.attn", "transformer_blocks.12.attn", "transformer_blocks.16.attn"]`
  - Apply to middle and later layers for best results
  - Can experiment with different layer combinations

### Standard Parameters

- **`guidance_scale`** (float, default: 3.5): Standard CFG guidance scale
- **`num_inference_steps`** (int, default: 20): Number of denoising steps
- **`seed`** (int, optional): Random seed for reproducibility

## 🔬 Technical Details

### NAG-PAG Attention Processor

The core innovation is in the `NAGPAGProcessor` class:

1. **Positive Branch**: Normal attention computation
2. **Negative Branch**: Identity matrix attention (PAG-style)
3. **Extrapolation**: `output = positive + scale * (positive - negative)`
4. **Normalization**: L1 normalization for stability (NAG-style)
5. **Blending**: Alpha blending for smooth transitions

### Architecture Compatibility

- ✅ **FLUX.1 [dev]**: Primary target architecture
- ✅ **Transformer-based models**: Should work with other DiT architectures
- ✅ **Memory efficient**: Uses forward hooks, minimal memory overhead
- ✅ **Training-free**: No additional training required

## 📊 Expected Benefits

### Quality Improvements
- 🎨 **Better structure**: Identity matrix guidance preserves structural coherence
- 🌈 **Enhanced details**: Feature extrapolation enhances fine details
- 🎯 **Stable output**: L1 normalization prevents divergence

### Practical Advantages
- 🚀 **No negative prompts needed**: Simpler prompt engineering
- ⚡ **Faster inference**: No need to run model twice (like CFG)
- 🔧 **Easy integration**: Drop-in replacement for existing pipelines

## 🛠️ Implementation Notes

### Error Handling
- Robust fallback mechanisms if attention hooks fail
- Architecture-agnostic layer targeting with alternatives
- Graceful degradation to original outputs

### Performance
- Forward hooks are applied only during inference
- Minimal computational overhead compared to CFG
- Memory-efficient implementation

### Debugging
- Built-in transformer structure analysis
- Detailed logging for troubleshooting
- Layer targeting validation

## 🔄 Comparison with Other Methods

| Method | Negative Guidance | Training Required | Computational Cost | Flexibility |
|--------|------------------|-------------------|-------------------|-------------|
| CFG | Negative prompts | No | 2x (dual forward pass) | High |
| PAG | Identity matrix | No | ~1.1x (attention hooks) | High |
| NAG | Feature extrapolation | No | ~1.2x (attention processing) | High |
| **NAG-PAG** | **Identity matrix + normalization** | **No** | **~1.1x** | **High** |

## 🎯 Best Practices

### Layer Selection
- Use middle to late transformer layers (8, 12, 16)
- Avoid very early layers (may affect basic structure)
- Experiment with different combinations for your use case

### Scale Tuning
- Start with `nag_pag_scale=3.0`
- Increase for stronger guidance effects
- Decrease if artifacts appear

### Prompt Engineering
- No negative prompts needed!
- Focus on positive descriptions
- NAG-PAG handles unwanted features automatically

## 📈 Future Enhancements

- [ ] Adaptive scaling based on inference step
- [ ] Layer-specific scaling parameters
- [ ] Integration with ControlNet
- [ ] Support for more transformer architectures
- [ ] Quantitative evaluation metrics

## 🤝 Contributing

This implementation is part of an applied ML project exploring novel attention guidance techniques. Feel free to experiment with different parameters and layer configurations!

## 📄 License

This implementation is provided for research and development purposes. Please ensure compliance with FLUX.1 [dev] model licensing terms. 