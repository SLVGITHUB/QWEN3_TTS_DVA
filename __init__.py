__version__ = "1.0.5"
__author__ = "DVA"
__description__ = "DVA Qwen3_TTS_Nodes for emotional clone speak"

try:
    import comfy.utils
    COMFYUI_AVAILABLE = True
except ImportError:
    COMFYUI_AVAILABLE = False
    print("[DVA Qwen-TTS] ComfyUI not available - running in standalone mode")
else:
    from .nodes import NODE_CLASS_MAPPINGS as imported_mappings
    from .nodes import NODE_DISPLAY_NAME_MAPPINGS as imported_display_names
    
    NODE_CLASS_MAPPINGS.update(imported_mappings)
    NODE_DISPLAY_NAME_MAPPINGS.update(imported_display_names)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "__version__", "__author__"]




