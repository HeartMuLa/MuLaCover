"""MuLaCover: controllable cover-song generation from audio or MIDI."""

from .configuration import MuLaCoverConfig
from .modeling import MuLaCover
from .pipeline import MuLaCoverGenConfig, MuLaCoverGenPipeline

__all__ = [
    "MuLaCover",
    "MuLaCoverConfig",
    "MuLaCoverGenConfig",
    "MuLaCoverGenPipeline",
]

__version__ = "0.1.0"
