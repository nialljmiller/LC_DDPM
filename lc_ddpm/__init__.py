"""lc-ddpm: conditional diffusion model for realistic variable-star light curves."""
__version__ = "2.0.0"

from .api import Generator, generate

__all__ = ["Generator", "generate", "__version__"]
