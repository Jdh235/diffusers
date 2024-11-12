# 🧨 Diffusers with BDIA-DDIM Sampler

This repository is a fork of the [Hugging Face Diffusers library](https://github.com/huggingface/diffusers) that implements the BDIA (Bi-directional Integration Approximation) technique from the ECCV 2024 paper ["Exact Diffusion Inversion via Bi-directional Integration Approximation"](https://arxiv.org/abs/placeholder) by Guoqiang Zhang, J. P. Lewis, and W. Bastiaan Kleijn.

## What's New

This fork adds the BDIA-DDIM sampler, which offers:
- Exact diffusion inversion through bi-directional integration approximation
- Improved sampling quality compared to traditional DDIM, particularly at lower timesteps
- Mathematically rigorous approach to diffusion model inversion

## Installation

### Local Installation
```bash
git clone https://github.com/YOUR_USERNAME/diffusers
cd diffusers
pip install -e .
```

### Google Colab Installation
```python
# Clone the repository
!git clone https://github.com/YOUR_USERNAME/diffusers
!pip install -e diffusers

# If you want to store the scheduler in Google Drive
from google.colab import drive
drive.mount('/content/drive', force_remount=True)
```

## Usage

### Basic Usage
The BDIA-DDIM sampler works particularly well with the Stable Diffusion 2.1 base model, especially at lower timesteps. Note that BDIA-DDIM requires deterministic sampling (eta=0) to ensure exact diffusion inversion:

```python
from diffusers import StableDiffusionPipeline
from scheduling_bdia_ddim import BDIA_DDIMScheduler
import torch

# Set model ID
model_id = "stabilityai/stable-diffusion-2-1-base"

# Initialize scheduler with BDIA parameters
# Note: eta must be 0 for BDIA-DDIM as it requires deterministic sampling
scheduler = BDIA_DDIMScheduler.from_pretrained(
    model_id, 
    subfolder="scheduler",
    eta=0,          # Must be 0 for deterministic sampling
    gamma=0.5       # BDIA gamma parameter (0.5 for low timesteps, 1.0 for ~100 timesteps)
)

# Initialize pipeline
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    scheduler=scheduler,
    torch_dtype=torch.float32
)
pipe = pipe.to("cuda")

# Generate image
prompt = "A man dressed for the snowy mountain looks at the camera"
image = pipe(
    prompt,
    num_inference_steps=10,    # BDIA-DDIM is particularly effective at lower timesteps
    guidance_scale=9.0,
    generator=torch.Generator("cuda").manual_seed(3)
).images[0]
```

### Google Colab Complete Example
```python
# Mount Google Drive
import sys
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

# Clone repository if not already done
!git clone https://github.com/YOUR_USERNAME/diffusers
!pip install -e diffusers

# Import required modules
from diffusers import StableDiffusionPipeline
from scheduling_bdia_ddim import BDIA_DDIMScheduler
import torch

# Set model ID
model_id = "stabilityai/stable-diffusion-2-1-base"

# Initialize scheduler
# Note: eta must be 0 for BDIA-DDIM as it requires deterministic sampling
scheduler = BDIA_DDIMScheduler.from_pretrained(
    model_id, 
    subfolder="scheduler", 
    eta=0,
    gamma=0.5  # 0.5 for low timesteps, 1.0 for ~100 timesteps
)

# Initialize pipeline
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    scheduler=scheduler,
    torch_dtype=torch.float32
)
pipe = pipe.to("cuda")

# Generate image
prompt = "A man dressed for the snowy mountain looks at the camera"
image = pipe(
    prompt,
    num_inference_steps=10,
    guidance_scale=9.0,
    generator=torch.Generator("cuda").manual_seed(3)
).images[0]

# Display image
image.show()
```

### Configuration Options

The BDIA-DDIM scheduler supports the following parameters:

- `gamma` (float): BDIA integration parameter
  - Use 0.5 for lower timesteps (recommended for steps < 20)
  - Use 1.0 for higher timesteps (~100)
  - Set to 0 to recreate standard DDIM behavior
- `num_inference_steps` (int): Number of denoising steps (BDIA-DDIM is particularly effective at lower steps)

Note: The `eta` parameter must always be set to 0 as BDIA-DDIM requires deterministic sampling to ensure exact diffusion inversion.

## Comparison with Standard DDIM

The BDIA technique offers several advantages over standard DDIM:
- Better performance at lower timesteps
- More accurate inversion of the diffusion process through deterministic sampling
- Better preservation of image details
- Theoretical guarantees for inversion accuracy

## Citation

If you use this implementation in your research, please cite both the original Diffusers library and the BDIA paper:

```bibtex
@inproceedings{zhang2024exact,
  title={Exact Diffusion Inversion via Bi-directional Integration Approximation},
  author={Zhang, Guoqiang and Lewis, J. P. and Kleijn, W. Bastiaan},
  booktitle={European Conference on Computer Vision},
  year={2024}
}

@misc{von-platen-etal-2022-diffusers,
  author = {Patrick von Platen and Suraj Patil and Anton Lozhkov and Pedro Cuenca and Nathan Lambert and Kashif Rasul and Mishig Davaadorj and Dhruv Nair and Sayak Paul and William Berman and Yiyi Xu and Steven Liu and Thomas Wolf},
  title = {Diffusers: State-of-the-art diffusion models},
  year = {2022},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/huggingface/diffusers}}
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The original Diffusers library team at Hugging Face
- The authors of the BDIA paper for their novel approach to diffusion inversion
- The open-source community for their continuous support and contributions
