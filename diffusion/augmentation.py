import torch


def add_gaussian_noise(image, strength=0.15):
    """Experimental diffusion-inspired noise augmentation.

    This utility is intentionally lightweight: it is not a trained DDPM and
    should be described as an experimental augmentation baseline.
    """
    strength = float(max(0.0, min(1.0, strength)))
    noise = torch.randn_like(image) * strength
    return (image + noise).clamp(0.0, 1.0)
