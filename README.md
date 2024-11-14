# 🧨 Diffusers with BDIA-DDIM Sampler

This repository is a fork of the [Hugging Face Diffusers library](https://github.com/huggingface/diffusers) that implements the BDIA (Bi-directional Integration Approximation) technique from the ECCV 2023 paper ["Exact Diffusion Inversion via Bi-directional Integration Approximation"](https://arxiv.org/abs/2307.10829) by Guoqiang Zhang, J. P. Lewis, and W. Bastiaan Kleijn.

## What's New

This fork adds the BDIA-DDIM sampler, which offers:
- Improved sampling quality compared to traditional DDIM, particularly at lower timesteps.
- Mathematically exact approach to diffusion model inversion, when used in 'round trip' BDIA-DDIM inversion.

## Installation

### Google Colab Installation
```python
# Clone the repository
!git clone https://github.com/Jdh235/diffusers
```

### Google Colab Complete Example
```python
# Full Colab Demo for Running Stable Diffusion with Custom BDIA-DDIM Scheduler

# Install dependencies (Uncomment if needed)
# !pip install torch transformers accelerate

# Clone the forked diffusers repository to Colab’s local storage
!git clone https://github.com/Jdh235/diffusers.git

# Uninstall the existing diffusers package (if any) to avoid conflicts
!pip uninstall diffusers -y

# Set the working directory to the cloned repository and update the Python path
%cd /content/diffusers/src
import sys
sys.path.append('/content/diffusers/src')

# Import necessary modules from the diffusers package
from diffusers import StableDiffusionPipeline, BDIA_DDIMScheduler
import torch

# Set model ID for the Stable Diffusion pipeline
model_id = "stabilityai/stable-diffusion-2-1-base"

# Set up the scheduler with parameters (adjust gamma to change behavior)
scheduler = BDIA_DDIMScheduler.from_pretrained(model_id, subfolder="scheduler", eta=0, gamma=0.5)

# Initialize the pipeline with the custom BDIA-DDIM scheduler
pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float32)
pipe = pipe.to("cuda")

# Define the prompt and generate the image
prompt = "A man dressed for the snowy mountain looks at the camera"
image = pipe(
    prompt,
    num_inference_steps=40,
    guidance_scale=9.0,
    eta=0,
    generator=torch.Generator("cuda").manual_seed(3),
).images[0]

#show image
image
```

### Configuration Options

The BDIA-DDIM scheduler supports the following parameters:

- `gamma` (float): BDIA integration parameter
  - Use 0.5 for lower timesteps (recommended for steps < 20)
  - Use 1.0 for higher timesteps (~100)
  - Set to 0 to recreate standard DDIM behavior
- `num_inference_steps` (int): Number of denoising steps (BDIA-DDIM is particularly effective at lower steps)

Note: The `eta` parameter must always be set to 0 as BDIA-DDIM requires deterministic sampling.

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

## Acknowledgments

- The original Diffusers library team at Hugging Face
- The authors of the BDIA paper for their novel approach to diffusion inversion
- The open-source community for their continuous support and contributions
