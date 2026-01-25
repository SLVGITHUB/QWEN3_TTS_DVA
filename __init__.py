# __init__.py

# === ComfyUI-Manager metadata ===
# This allows the node to appear in ComfyUI Manager with description and author info
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

__version__ = "1.0.0"
__author__ = "DVA"
__description__ = "DVA Qwen-TTS: High-quality multilingual text-to-speech with voice cloning and emotion control."

# Only import if running inside ComfyUI
try:
    import comfy.utils
except ImportError:
    pass
else:
    from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
    __all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]