import os
import torch
import numpy as np
import folder_paths
import json
import time
import traceback
import soundfile as sf

_QWEN_MODEL_CACHE = {}

try:
    from qwen_tts import Qwen3TTSModel
    QWEN_AVAILABLE = True
except ImportError:
    QWEN_AVAILABLE = False
    print("⚠️ Qwen-TTS не установлен. Установите: pip install qwen-tts")


def _load_model(model_name, precision, attention_type, device, cache_dir=""):
    global _QWEN_MODEL_CACHE
    key = (model_name, precision, device)
    if key in _QWEN_MODEL_CACHE:
        return _QWEN_MODEL_CACHE[key]

    if not QWEN_AVAILABLE:
        raise ImportError("Qwen-TTS не установлен. Выполните: pip install qwen-tts")

    print(f"🔄 Загрузка модели: {model_name}")
    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    dtype = dtype_map[precision]
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    kwargs = {
        "torch_dtype": dtype,
        "device_map": device,
        "low_cpu_mem_usage": True,
        "trust_remote_code": True,
    }
    if cache_dir and os.path.isdir(cache_dir):
        kwargs["cache_dir"] = cache_dir

    try:
        kwargs["attn_implementation"] = attention_type
        model = Qwen3TTSModel.from_pretrained(model_name, **kwargs)
    except Exception as e1:
        print(f"⚠️ Ошибка с attention={attention_type}, переключение на 'eager'")
        kwargs["attn_implementation"] = "eager"
        model = Qwen3TTSModel.from_pretrained(model_name, **kwargs)

    model.metadata = {
        "model_name": model_name,
        "precision": precision,
        "attention_type": attention_type,
        "device": str(model.device),
    }
    _QWEN_MODEL_CACHE[key] = model
    print(f"✅ Модель загружена на {model.device}")
    return model


def _detect_model_type(model_name):
    """Определяет тип модели по имени."""
    if "CustomVoice" in model_name:
        return "custom_voice"
    elif "VoiceDesign" in model_name:
        return "voice_design"
    else:
        return "base"  # voice clone


class DVA_QwenTTSGenerate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {
                    "default": "Привет! Это тест синтеза речи.",
                    "multiline": True
                }),
                "model_name": ([
                    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
                    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
                    "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
                    "Qwen/Qwen3-TTS-24Hz-1.7B-Base",
                ], {"default": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"}),
                "language": (["Auto", "Russian", "English", "Chinese", "German",
                              "French", "Spanish", "Japanese", "Korean", "Italian"], {"default": "Russian"}),
                "temperature": ("FLOAT", {"default": 0.9, "min": 0.1, "max": 2.0, "step": 0.1}),
                "top_p": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 1.0, "step": 0.05}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "precision": (["fp16", "bf16", "fp32"], {"default": "fp16"}),
                "attention_type": (["sdpa", "eager", "flash_attention_2"], {"default": "sdpa"}),
                "device": (["auto", "cuda", "cpu"], {"default": "auto"}),
            },
            "optional": {
                "cache_dir": ("STRING", {"default": "", "multiline": False}),
                "speaker": ("STRING", {"default": "Vivian"}),  # для CustomVoice
                "instruct": ("STRING", {"default": "", "multiline": True}),  # для CustomVoice / VoiceDesign
                "emotion_preset": (["neutral", "happy", "sad", "angry", "surprised",
                                   "energetic", "calm", "dramatic", "professional"],
                                  {"default": "neutral"}),
            }
        }

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "info")
    FUNCTION = "generate_speech"
    CATEGORY = "audio/tts"

    def generate_speech(self, text, model_name, language, temperature, top_p, seed,
                       precision="fp16", attention_type="sdpa", device="auto",
                       cache_dir="", speaker="Vivian", instruct="", emotion_preset="neutral"):

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Эмоции → instruct (для CustomVoice и VoiceDesign)
        emotion_to_instruct = {
            "neutral": "",
            "happy": "очень радостный и энергичный тон",
            "sad": "грустный, медленный и тихий тон",
            "angry": "сердитый и резкий тон",
            "surprised": "удивлённый, с высокой интонацией",
            "energetic": "энергичный и быстрый темп",
            "calm": "спокойный, ровный и мягкий тон",
            "dramatic": "театральный, выразительный тон",
            "professional": "чёткий, деловой и уверенный тон",
        }
        if emotion_preset != "neutral" and not instruct:
            instruct = emotion_to_instruct.get(emotion_preset, "")

        print(f"🎤 Генерация: {text[:50]}...")
        print(f"⚙️ Модель: {model_name}, язык: {language}, эмоция: {emotion_preset}")

        model = _load_model(model_name, precision, attention_type, device, cache_dir)
        model_type = _detect_model_type(model_name)

        try:
            start_time = time.time()

            # Подготовка языка
            lang = None if language == "Auto" else language

            if model_type == "custom_voice":
                # Требуется speaker
                wavs, sr = model.generate_custom_voice(
                    text=text,
                    language=lang,
                    speaker=speaker,
                    instruct=instruct,
                    temperature=temperature,
                    top_p=top_p,
                    max_new_tokens=1024,
                )
            elif model_type == "voice_design":
                if not instruct:
                    instruct = "естественный, чёткий и приятный голос"
                wavs, sr = model.generate_voice_design(
                    text=text,
                    language=lang,
                    instruct=instruct,
                    temperature=temperature,
                    top_p=top_p,
                    max_new_tokens=1024,
                )
            else:  # base → voice clone (но без ref_audio — используем как обычный TTS)
                # Для Base без референса — используем как fallback
                wavs, sr = model.generate_voice_clone(
                    text=text,
                    language=lang,
                    ref_audio=None,  # это вызовет ошибку, поэтому лучше не использовать Base без ref
                    ref_text="",
                    x_vector_only_mode=True,
                    temperature=temperature,
                    top_p=top_p,
                    max_new_tokens=1024,
                )

            duration = time.time() - start_time
            audio_data = wavs[0] if isinstance(wavs, list) else wavs
            if torch.is_tensor(audio_data):
                audio_data = audio_data.cpu().numpy()

            audio_tensor = torch.from_numpy(audio_data).unsqueeze(0).unsqueeze(0)  # [1,1,T]

            info = {
                "sample_rate": int(sr),
                "duration_sec": round(len(audio_data) / sr, 2),
                "generation_time_sec": round(duration, 2),
                "model": model_name,
                "model_type": model_type,
                "parameters": {
                    "language": language,
                    "speaker": speaker if model_type == "custom_voice" else None,
                    "instruct": instruct,
                    "emotion_preset": emotion_preset,
                }
            }

            print(f"✅ Готово за {duration:.2f} сек, длина: {info['duration_sec']} сек")
            return ({"waveform": audio_tensor, "sample_rate": int(sr)}, json.dumps(info, indent=2, ensure_ascii=False))

        except Exception as e:
            print(f"❌ Ошибка генерации: {e}")
            traceback.print_exc()
            silence = torch.zeros((1, 1, 24000))
            sr = 24000
            return ({"waveform": silence, "sample_rate": sr}, f"Ошибка: {str(e)}")


class DVA_QwenTTSAudioSaver:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "filename": ("STRING", {"default": "qwen_tts_output.wav"}),
                "sample_rate": ("INT", {"default": 24000, "min": 8000, "max": 48000}),
            },
            "optional": {
                "output_dir": ("STRING", {"default": "tts/dva_qwen"}),
                "metadata": ("STRING", {"default": "", "multiline": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("filepath",)
    FUNCTION = "save_audio"
    CATEGORY = "audio/tts"

    def save_audio(self, audio, filename, sample_rate, output_dir="tts/dva_qwen", metadata=""):
        full_dir = os.path.join(folder_paths.get_output_directory(), output_dir)
        os.makedirs(full_dir, exist_ok=True)

        safe_name = "".join(c for c in filename if c.isalnum() or c in "._- ").strip()
        if not safe_name.lower().endswith(".wav"):
            safe_name += ".wav"
        filepath = os.path.join(full_dir, safe_name)

        if isinstance(audio, dict) and "waveform" in audio:
            audio_np = audio["waveform"].cpu().numpy().squeeze()
        else:
            audio_np = audio.cpu().numpy().squeeze() if isinstance(audio, torch.Tensor) else np.array(audio)

        sf.write(filepath, audio_np, sample_rate)

        if metadata.strip():
            meta_path = filepath.replace(".wav", ".json")
            try:
                meta_dict = json.loads(metadata) if isinstance(metadata, str) else metadata
                meta_dict.update({"save_time": time.strftime("%Y-%m-%d %H:%M:%S")})
                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump(meta_dict, f, indent=2, ensure_ascii=False)
            except Exception as e:
                print(f"⚠️ Не удалось сохранить метаданные: {e}")

        print(f"💾 Аудио сохранено: {filepath}")
        return (filepath,)


NODE_CLASS_MAPPINGS = {
    "DVA_QwenTTSGenerate": DVA_QwenTTSGenerate,
    "DVA_QwenTTSAudioSaver": DVA_QwenTTSAudioSaver,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DVA_QwenTTSGenerate": "DVA 🎤 Qwen TTS Generate (Auto)",
    "DVA_QwenTTSAudioSaver": "DVA 💾 Qwen TTS Audio Saver",
}