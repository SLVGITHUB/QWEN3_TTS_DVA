__version__ = "1.0.0"
__author__ = "DVA"
__description__ = "DVA Qwen-TTS: High-quality multilingual text-to-speech with voice cloning and emotion control."

# Инициализируем пустыми значениями по умолчанию
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# Пытаемся импортировать ноды только если ComfyUI доступен
try:
    import comfy.utils
    COMFYUI_AVAILABLE = True
except ImportError:
    COMFYUI_AVAILABLE = False
    print("[DVA Qwen-TTS] ComfyUI not available - running in standalone mode")
else:
    # Только если ComfyUI доступен, импортируем ноды
    from .nodes import NODE_CLASS_MAPPINGS as imported_mappings
    from .nodes import NODE_DISPLAY_NAME_MAPPINGS as imported_display_names
    
    # Обновляем маппинги
    NODE_CLASS_MAPPINGS.update(imported_mappings)
    NODE_DISPLAY_NAME_MAPPINGS.update(imported_display_names)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "__version__", "__author__"]
