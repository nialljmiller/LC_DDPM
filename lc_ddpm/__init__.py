"""lc-ddpm: conditional diffusion model for realistic variable-star light curves."""
__version__ = "2.0.0"

from .api import Generator, generate
from .api_v3 import GeneratorV3

__all__ = ["Generator", "GeneratorV3", "generate", "__version__"]
