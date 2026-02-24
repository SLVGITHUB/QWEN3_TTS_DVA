import os
import torch
import numpy as np
import folder_paths
import json
import time
import traceback
import tempfile
import soundfile as sf
import sounddevice as sd
import re
import uuid
import pathlib
import posixpath
import hashlib
from typing import List, Dict, Any, Optional, Union

# Для работы с MP3 и другими форматами
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    print("⚠️ librosa не установлен. Установите: pip install librosa")
    print("   librosa нужен для конвертации MP3, OGG и других форматов")

try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False
    print("⚠️ pydub не установлен. Установите: pip install pydub")
    print("   pydub нужен как запасной вариант для конвертации форматов")

# Попробуем импортировать qwen_tts
try:
    from qwen_tts import Qwen3TTSModel
    QWEN_AVAILABLE = True
except ImportError:
    QWEN_AVAILABLE = False
    print("⚠️ Qwen-TTS не установлен. Установите: pip install qwen-tts")

# Глобальная переменная для хранения пользовательского пути временных файлов
CUSTOM_TEMP_DIR = None

def set_temp_directory(path: str):
    """Устанавливает пользовательскую директорию для временных файлов"""
    global CUSTOM_TEMP_DIR
    if path and isinstance(path, str) and path.strip():
        try:
            os.makedirs(path, exist_ok=True)
            test_file = os.path.join(path, "test_write.tmp")
            with open(test_file, 'w') as f:
                f.write("test")
            os.remove(test_file)
            CUSTOM_TEMP_DIR = path
            print(f"📁 Установлена пользовательская временная директория: {path}")
        except Exception as e:
            print(f"⚠️ Не удалось использовать пользовательскую директорию {path}: {e}")
            CUSTOM_TEMP_DIR = None

def get_temp_directory():
    """Возвращает текущую временную директорию"""
    if CUSTOM_TEMP_DIR and os.path.exists(CUSTOM_TEMP_DIR):
        return CUSTOM_TEMP_DIR
    return tempfile.gettempdir()

def _detect_model_type(model_name):
    """Безопасное определение типа модели"""
    if not isinstance(model_name, str):
        return "base"
    model_name_lower = model_name.lower()
    if "customvoice" in model_name_lower:
        return "custom_voice"
    elif "voicedesign" in model_name_lower or "voice_design" in model_name_lower:
        return "voice_design"
    else:
        return "base"

def _sanitize_filename(filename, max_length=100):
    """Безопасное создание имени файла без path traversal"""
    if not filename:
        return "output.wav"
    base_name = os.path.basename(str(filename))
    sanitized = re.sub(r'[^\w\s\.\-]', '_', base_name)
    sanitized = re.sub(r'_+', '_', sanitized)
    sanitized = sanitized.strip('. ')
    if len(sanitized) > max_length:
        name, ext = os.path.splitext(sanitized)
        sanitized = name[:max_length - len(ext)] + ext
    if not sanitized or sanitized in ['.', '..']:
        sanitized = "output"
    if not sanitized.lower().endswith('.wav'):
        sanitized += '.wav'
    return sanitized

def _safe_json_loads(json_str, max_size=10000):
    """Безопасная загрузка JSON с ограничениями"""
    if not json_str or not isinstance(json_str, str):
        return {}
    if len(json_str) > max_size:
        json_str = json_str[:max_size]
    try:
        parsed = json.loads(json_str)
        if not isinstance(parsed, (dict, list)):
            return {"data": str(parsed)[:500]}
        
        def limit_structure(obj, depth=0, max_depth=10, max_items=100):
            if depth > max_depth:
                return "[max depth reached]"
            if isinstance(obj, dict):
                items = list(obj.items())[:max_items]
                return {str(k)[:100]: limit_structure(v, depth+1, max_depth, max_items) 
                       for k, v in items}
            elif isinstance(obj, list):
                return [limit_structure(item, depth+1, max_depth, max_items) 
                       for item in obj[:max_items]]
            elif isinstance(obj, str):
                 return obj[:500]
            elif isinstance(obj, (int, float, bool, type(None))):
                return obj
            else:
                return str(obj)[:200]
        
        return limit_structure(parsed)
    except (json.JSONDecodeError, TypeError, ValueError) as e:
        return {"error": "invalid_json", "message": str(e)[:200]}

def _validate_path(path, allowed_base=None):
    """Проверка безопасности пути"""
    if not path or not isinstance(path, str):
        return False, "Путь должен быть строкой"
    normalized = os.path.normpath(path)
    if '..' in normalized.split(os.sep):
        return False, "Путь содержит '..' (path traversal)"
    if os.path.isabs(path) and allowed_base:
        try:
            common = os.path.commonpath([normalized, allowed_base])
            if common != allowed_base:
                return False, "Путь находится вне разрешенной директории"
        except ValueError:
            return False, "Недопустимый путь"
    if len(path) > 500:
        return False, "Слишком длинный путь"
    return True, normalized

def _resample_audio(audio_np, orig_sr, target_sr):
    """Ресемплинг аудио с использованием линейной интерполяции"""
    if orig_sr == target_sr:
        return audio_np
    duration = len(audio_np) / orig_sr
    target_len = int(duration * target_sr)
    if target_len <= 1:
        print(f"⚠️ Слишком короткое аудио для ресемплинга")
        return audio_np
    orig_times = np.linspace(0, duration, len(audio_np))
    target_times = np.linspace(0, duration, target_len)
    resampled = np.interp(target_times, orig_times, audio_np)
    return resampled.astype(np.float32)

def _prepare_audio_for_qwen(audio_data, original_sr=None, target_sr=24000):
    """
    Подготавливает аудио для Qwen TTS модели:
    - Принимает ЛЮБЫЕ форматы: mp3, flac, wav, ogg
    - Конвертирует стерео в моно
    - Ресемплинг в target_sr (24000 Hz)
    - Конвертирует в float32 в диапазоне [-1, 1]
    """
    print(f"🎧 Подготовка аудио для Qwen модели (целевой SR: {target_sr})...")

    audio_np = None
    file_sr = original_sr

    # Случай 1: аудио как путь к файлу (MP3, WAV, FLAC, OGG и т.д.)
    if isinstance(audio_data, str) and os.path.exists(audio_data):
        file_ext = os.path.splitext(audio_data)[1].lower()
        print(f"📁 Загрузка файла: {audio_data} (формат: {file_ext})")
        
        # ПРИОРИТЕТ 1: librosa (поддерживает всё через ffmpeg)
        if LIBROSA_AVAILABLE:
            try:
                # sr=target_sr делает ресемплинг сразу, mono=True делает моно
                audio_np, file_sr = librosa.load(audio_data, sr=target_sr, mono=True)
                print(f"✅ librosa загрузил: {len(audio_np)/target_sr:.2f} сек, {target_sr}Hz, моно")
            except Exception as e:
                print(f"⚠️ Ошибка librosa: {e}. Пробуем другие методы...")
                audio_np = None
        
        # ПРИОРИТЕТ 2: pydub (если librosa не сработал)
        if audio_np is None and PYDUB_AVAILABLE:
            try:
                audio = AudioSegment.from_file(audio_data)
                if audio.channels > 1:
                    print(f"🎵 Конвертация стерео в моно (pydub)")
                    audio = audio.set_channels(1)
                if audio.frame_rate != target_sr:
                    print(f"🔄 Ресемплинг {audio.frame_rate}Hz -> {target_sr}Hz (pydub)")
                    audio = audio.set_frame_rate(target_sr)
                
                samples = np.array(audio.get_array_of_samples())
                if audio.sample_width == 2:  # 16-bit
                    audio_np = samples.astype(np.float32) / 32768.0
                elif audio.sample_width == 4:  # 32-bit
                    audio_np = samples.astype(np.float32) / 2147483648.0
                else:
                    audio_np = samples.astype(np.float32)
                    max_val = np.max(np.abs(audio_np))
                    if max_val > 0:
                        audio_np = audio_np / max_val
                file_sr = target_sr
                print(f"✅ pydub загрузил: {len(audio_np)/target_sr:.2f} сек")
            except Exception as e:
                print(f"⚠️ Ошибка pydub: {e}")
                audio_np = None
        
        # ПРИОРИТЕТ 3: soundfile (только для WAV, FLAC, OGG - MP3 может не работать без libsndfile mp3 patch)
        if audio_np is None:
            try:
                # Пытаемся прочитать напрямую
                data, sr = sf.read(audio_data)
                audio_np = data
                file_sr = sr
                print(f"✅ soundfile загрузил: SR={file_sr}Hz, форма={audio_np.shape}")
            except Exception as e:
                raise ValueError(f"Не удалось прочитать аудио файл {audio_data} ни одним из методов (librosa, pydub, soundfile). Ошибка: {e}")

    # Случай 2: аудио как dict от ComfyUI
    elif isinstance(audio_data, dict) and "waveform" in audio_data:
        waveform = audio_data["waveform"]
        file_sr = audio_data.get("sample_rate", original_sr)
        
        if torch.is_tensor(waveform):
            audio_np = waveform.detach().cpu().numpy()
        else:
            audio_np = np.array(waveform)
        print(f"📊 ComfyUI аудио: форма={audio_np.shape}, SR={file_sr}Hz")

    # Случай 3: аудио как torch тензор
    elif torch.is_tensor(audio_data):
        audio_np = audio_data.detach().cpu().numpy()
        print(f"📊 Torch тензор: форма={audio_np.shape}")

    # Случай 4: аудио как numpy массив
    elif isinstance(audio_data, np.ndarray):
        audio_np = audio_data.copy()
        print(f"📊 NumPy массив: форма={audio_np.shape}, dtype={audio_np.dtype}")

    # Случай 5: аудио как список/кортеж
    elif isinstance(audio_data, (list, tuple)):
        audio_np = np.array(audio_data, dtype=np.float32)
        print(f"📊 Список/кортеж: форма={audio_np.shape}")

    else:
        raise ValueError(f"Неподдерживаемый формат аудио: {type(audio_data)}")

    # === ОБЯЗАТЕЛЬНАЯ НОРМАЛИЗАЦИЯ ДАННЫХ ===
    
    # 1. Убираем лишние размерности
    audio_np = audio_np.squeeze()

    # 2. Обработка стерео -> моно (если вдруг осталось)
    if audio_np.ndim > 1:
        if audio_np.shape[0] == 2 and audio_np.shape[1] > 2:
            print("🎵 Конвертация стерео в моно (среднее по каналам) - формат [2, samples]")
            audio_np = np.mean(audio_np, axis=0)
        elif audio_np.shape[1] == 2 and audio_np.shape[0] > 2:
            print("🎵 Конвертация стерео в моно (среднее по каналам) - формат [samples, 2]")
            audio_np = np.mean(audio_np, axis=1)
        else:
            print(f"⚠️ Необычная размерность {audio_np.shape}, применяем flatten")
            audio_np = audio_np.flatten()

    # 3. Приводим к float32
    if audio_np.dtype != np.float32:
        print(f"🔄 Конвертация {audio_np.dtype} в float32")
        audio_np = audio_np.astype(np.float32)

    # 4. Нормализуем в диапазон [-1, 1]
    if len(audio_np) > 0:
        max_val = np.max(np.abs(audio_np))
        if max_val > 1.0:
            print(f"🔄 Нормализация: max={max_val:.3f} -> 1.0")
            audio_np = audio_np / max_val
        elif max_val < 0.01 and max_val > 0:
            print(f"⚠️ Аудио очень тихое (max={max_val:.3f}), усиливаем")
            audio_np = audio_np * (0.5 / max_val)

    # 5. Ресемплинг если нужно (если не сделал librosa)
    if file_sr and file_sr != target_sr:
        print(f"🔄 Ресемплинг: {file_sr}Hz -> {target_sr}Hz")
        audio_np = _resample_audio(audio_np, file_sr, target_sr)
    elif not file_sr:
        print(f"⚠️ Частота дискретизации неизвестна, предполагаем {target_sr}Hz")
        file_sr = target_sr

    # 6. Обрезаем до 30 секунд (максимум для клонирования)
    max_samples = 30 * target_sr
    if len(audio_np) > max_samples:
        print(f"✂️ Обрезка аудио: {len(audio_np)/target_sr:.1f}сек -> 30сек")
        audio_np = audio_np[:max_samples]

    # 7. Финальная проверка на тишину
    if np.max(np.abs(audio_np)) < 0.001:
        print("⚠️ Внимание: аудио почти тихое, добавляем тестовый сигнал")
        t = np.linspace(0, 3, 3 * target_sr)
        test_signal = 0.3 * np.sin(2 * np.pi * 440 * t)
        audio_np = test_signal

    print(f"✅ Готово: {len(audio_np)/target_sr:.2f} сек, {target_sr}Hz, моно, float32, диапазон [{audio_np.min():.3f}, {audio_np.max():.3f}]")

    return audio_np, target_sr

def _create_safe_temp_file(suffix='.wav', data=None, sample_rate=24000, custom_dir=None):
    """
    Создание безопасного временного файла с гарантированным форматом для Qwen.
    Гарантирует: WAV, моно, 24000Hz, 16-bit PCM.
    """
    # Определяем директорию
    if custom_dir and os.path.exists(custom_dir):
        temp_dir = custom_dir
    elif CUSTOM_TEMP_DIR and os.path.exists(CUSTOM_TEMP_DIR):
        temp_dir = CUSTOM_TEMP_DIR
    else:
        temp_dir = tempfile.gettempdir()

    temp_filename = f"qwen_tts_{uuid.uuid4().hex}_{int(time.time())}{suffix}"
    temp_path = os.path.join(temp_dir, temp_filename)

    print(f"📝 Создание временного файла: {temp_path}")

    if data is not None:
        try:
            # Если данные уже подготовлены функцией _prepare_audio_for_qwen (кортеж)
            if isinstance(data, tuple) and len(data) == 2:
                audio_np, sr = data
            else:
                # Иначе пытаемся подготовить автоматически
                audio_np, sr = _prepare_audio_for_qwen(data, target_sr=sample_rate)
            
            if len(audio_np) == 0:
                raise ValueError("Пустые аудио данные")
            
            # Критически важно: убедиться, что массив одномерный (моно)
            if audio_np.ndim != 1:
                print(f"⚠️ Исправление размерности с {audio_np.shape} на (N,)")
                audio_np = audio_np.flatten()
            
            # Клиппинг диапазона
            audio_np = np.clip(audio_np, -1.0, 1.0)
            
            # Явная запись с указанием формата и подтипа
            # subtype='PCM_16' гарантирует совместимость
            sf.write(temp_path, audio_np, sr, format='WAV', subtype='PCM_16')
            
            if not os.path.exists(temp_path):
                raise ValueError("Файл не был создан после записи")
            
            file_size = os.path.getsize(temp_path)
            print(f"✅ Временный файл создан: {temp_path} ({file_size/1024:.2f} KB)")
            return temp_path
            
        except Exception as e:
            print(f"❌ Ошибка создания временного файла: {e}")
            traceback.print_exc()
            
            # Альтернативный путь (fallback)
            alt_dir = os.path.join(os.path.dirname(__file__), "temp_audio")
            try:
                os.makedirs(alt_dir, exist_ok=True)
                alt_path = os.path.join(alt_dir, temp_filename)
                print(f"📝 Пробуем альтернативный путь: {alt_path}")
                
                t = np.linspace(0, 3, 3 * sample_rate)
                test_signal = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
                
                sf.write(alt_path, test_signal, sample_rate, format='WAV', subtype='PCM_16')
                print(f"✅ Альтернативный временный файл создан: {alt_path}")
                return alt_path
            except Exception as e2:
                raise ValueError(f"Не удалось создать временный файл даже в альтернативном пути: {e2}")

    return temp_path

class QwenTTSConfig:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "temp_directory": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Путь для временных файлов (оставьте пустым для системной temp)"
                }),
            },
            "optional": {
                "create_dir": (["yes", "no"], {"default": "yes"}),
            }
        }
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("temp_path",)
    FUNCTION = "configure"
    CATEGORY = "audio/tts"

    def configure(self, temp_directory, create_dir="yes"):
        if temp_directory and temp_directory.strip():
            clean_path = temp_directory.strip()
            clean_path = re.sub(r'\.\./', '', clean_path)
            clean_path = re.sub(r'\.\.\\', '', clean_path)
            if create_dir == "yes":
                try:
                    os.makedirs(clean_path, exist_ok=True)
                    print(f"📁 Создана директория: {clean_path}")
                except Exception as e:
                    print(f"⚠️ Не удалось создать директорию: {e}")
                    return (" ",)
            set_temp_directory(clean_path)
            return (clean_path,)
        set_temp_directory(None)
        return (" ",)

class QwenTTSModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": ([
                    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
                    "Qwen/Qwen3-TTS-24Hz-1.7B-Base",
                    "Qwen/Qwen3-TTS-12Hz-1.7B-Instruct",
                    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
                    "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
                ],),
                "precision": (["fp16", "bf16", "fp32"], {"default": "fp16"}),
                "attention_type": (["sdpa", "eager", "flash_attention_2"], {"default": "sdpa"}),
                "device": (["auto", "cuda", "cpu"], {"default": "auto"}),
            },
            "optional": {
                "cache_dir": ("STRING", {"default": "", "multiline": False}),
            }
        }
    RETURN_TYPES = ("QWEN_TTS_MODEL",)
    RETURN_NAMES = ("qwen_model",)
    FUNCTION = "load_model"
    CATEGORY = "audio/tts"

    def load_model(self, model_name, precision, attention_type, device, cache_dir=" "):
        if not QWEN_AVAILABLE:
            raise ImportError("Qwen-TTS не установлен. Установите: pip install qwen-tts")
        
        print(f"🔄 Загрузка модели TTS: {model_name}")
        if not isinstance(model_name, str):
            raise ValueError("model_name должен быть строкой")
        
        dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
        dtype = dtype_map.get(precision, torch.float16)
        
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        try:
            kwargs = {
                "torch_dtype": dtype,
                "device_map": device,
                "low_cpu_mem_usage": True,
                "trust_remote_code": False,
            }
            trusted_models = [
                "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
                "Qwen/Qwen3-TTS-24Hz-1.7B-Base",
                "Qwen/Qwen3-TTS-12Hz-1.7B-Instruct",
                "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
                "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
            ]
            if model_name in trusted_models:
                kwargs["trust_remote_code"] = True
                print(f"✅ Модель {model_name} в списке доверенных")
            
            if cache_dir and isinstance(cache_dir, str) and cache_dir.strip():
                cache_dir = cache_dir.strip()
                is_valid, validated_path = _validate_path(cache_dir)
                if is_valid and os.path.exists(os.path.dirname(validated_path)):
                    kwargs["cache_dir"] = validated_path
            
            try:
                kwargs["attn_implementation"] = attention_type
                model = Qwen3TTSModel.from_pretrained(model_name, **kwargs)
            except Exception as e:
                print(f"⚠️ Fallback на eager attention: {e}")
                kwargs["attn_implementation"] = "eager"
                model = Qwen3TTSModel.from_pretrained(model_name, **kwargs)
            
            print(f"✅ Модель загружена на {model.device}")
            model.metadata = {
                "model_name": model_name,
                "precision": precision,
                "attention_type": attention_type,
                "device": str(model.device),
                "model_type": _detect_model_type(model_name),
                "trusted": model_name in trusted_models,
            }
            return (model,)
        except Exception as e:
            print(f"❌ Ошибка загрузки модели: {e}")
            traceback.print_exc()
            raise e

class QwenTTSGenerate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "qwen_model": ("QWEN_TTS_MODEL",),
                "text": ("STRING", {"default": "Привет! Это тест синтеза речи.", "multiline": True}),
                "language": (["Auto", "Russian", "English", "Chinese", "German", "French", "Spanish", "Japanese", "Korean"], {"default": "Russian"}),
                "temperature": ("FLOAT", {"default": 0.9, "min": 0.1, "max": 2.0, "step": 0.1}),
                "top_p": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 1.0, "step": 0.05}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "speaker": ("STRING", {"default": "Vivian"}),
                "instruct": ("STRING", {"default": " ", "multiline": True}),
                "emotion_preset": (["neutral", "happy", "sad", "angry", "surprised", "energetic", "calm", "dramatic", "professional"], {"default": "neutral"}),
            }
        }
    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "info")
    FUNCTION = "generate_speech"
    CATEGORY = "audio/tts"

    def generate_speech(self, qwen_model, text, language, temperature, top_p, seed, speaker="Vivian", instruct=" ", emotion_preset="neutral"):
        try:
            seed = int(seed) & 0xFFFFFFFF
        except (ValueError, TypeError):
            seed = 0
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        if not isinstance(text, str): text = str(text)
        text = text.strip()[:5000]
        if speaker and isinstance(speaker, str):
            speaker = re.sub(r'[^\w\s\-]', '', speaker.strip())[:50]
        if instruct and isinstance(instruct, str):
            instruct = instruct.strip()[:1000]
        
        emotion_to_instruct = {
            "neutral": "", "happy": "радостный и энергичный тон", "sad": "грустный и медленный тон",
            "angry": "сердитый и резкий тон", "surprised": "удивлённый, с высокой интонацией",
            "energetic": "быстрый и энергичный темп", "calm": "спокойный и мягкий тон",
            "dramatic": "театральный и выразительный тон", "professional": "чёткий и деловой тон",
        }
        if emotion_preset != "neutral" and not instruct:
            instruct = emotion_to_instruct.get(emotion_preset, "")
        
        model_type = qwen_model.metadata.get("model_type", "base")
        lang = None if language == "Auto" else language
        
        print(f"🎤 Генерация: {text[:50]}...")
        try:
            start_time = time.time()
            safe_temperature = max(0.1, min(2.0, float(temperature)))
            safe_top_p = max(0.1, min(1.0, float(top_p)))
            
            if model_type == "custom_voice":
                if not speaker: speaker = "Vivian"
                wavs, sr = qwen_model.generate_custom_voice(text=text, language=lang, speaker=speaker, instruct=instruct, temperature=safe_temperature, top_p=safe_top_p, max_new_tokens=1024)
            elif model_type == "voice_design":
                if not instruct: instruct = "естественный и чёткий голос"
                wavs, sr = qwen_model.generate_voice_design(text=text, language=lang, instruct=instruct, temperature=safe_temperature, top_p=safe_top_p, max_new_tokens=1024)
            else:
                if not instruct: instruct = "нейтральный тон"
                wavs, sr = qwen_model.generate_voice_design(text=text, language=lang, instruct=instruct, temperature=safe_temperature, top_p=safe_top_p, max_new_tokens=1024)
            
            duration = time.time() - start_time
            audio_data = wavs[0] if isinstance(wavs, list) else wavs
            if torch.is_tensor(audio_data):
                audio_data = audio_data.detach().cpu().numpy()
            
            max_samples = 10 * 60 * sr
            if len(audio_data) > max_samples:
                audio_data = audio_data[:max_samples]
            
            audio_tensor = torch.from_numpy(audio_data).unsqueeze(0).unsqueeze(0)
            info = {
                "sample_rate": int(sr), "duration_sec": round(len(audio_data) / sr, 2),
                "generation_time_sec": round(duration, 2), "model": qwen_model.metadata.get("model_name", "unknown"),
                "model_type": model_type, "parameters": {"language": language, "speaker": speaker if model_type == "custom_voice" else "default", "instruct": instruct[:100] if instruct else "", "emotion_preset": emotion_preset, "text_length": len(text)},
                "security": {"trusted_model": qwen_model.metadata.get("trusted", False), "input_sanitized": True}
            }
            print(f"✅ Готово за {duration:.2f} сек")
            return ({"waveform": audio_tensor, "sample_rate": int(sr)}, json.dumps(info, indent=2, ensure_ascii=False))
        except Exception as e:
            print(f"❌ Ошибка генерации: {e}")
            traceback.print_exc()
            silence = torch.zeros((1, 1, 24000))
            return ({"waveform": silence, "sample_rate": 24000}, json.dumps({"error": str(e)[:200]}, indent=2))

class QwenTTSVoiceClone:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "qwen_model": ("QWEN_TTS_MODEL",),
                "text": ("STRING", {"default": "Привет! Это клонированный голос.", "multiline": True}),
                "ref_audio": ("AUDIO",),
                "ref_text": ("STRING", {"default": "Это текст из референсного аудио.", "multiline": True}),
                "language": (["Russian", "English", "Chinese", "Auto"], {"default": "Russian"}),
                "clone_mode": (["x_vector_only", "icl"], {"default": "x_vector_only"}),
                "temperature": ("FLOAT", {"default": 0.8, "min": 0.1, "max": 2.0, "step": 0.1}),
            },
            "optional": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "output_prefix": ("STRING", {"default": "clone_"}),
                "temp_directory": ("STRING", {"default": "", "multiline": False}),
            }
        }
    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "info")
    FUNCTION = "clone_voice"
    CATEGORY = "audio/tts"

    def clone_voice(self, qwen_model, text, ref_audio, ref_text, language, clone_mode, temperature, seed=0, output_prefix="clone_", temp_directory=" "):
        model_type = qwen_model.metadata.get("model_type", "base")
        if model_type != "base":
            raise ValueError("Клонирование работает только с моделями типа '-Base'")
        
        try: seed = int(seed) & 0xFFFFFFFF
        except (ValueError, TypeError): seed = 0
        torch.manual_seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
        
        if not isinstance(text, str): text = str(text)
        text = text.strip()[:5000]
        if not isinstance(ref_text, str): ref_text = str(ref_text)
        ref_text = ref_text.strip()[:5000]
        if output_prefix and isinstance(output_prefix, str):
            output_prefix = _sanitize_filename(output_prefix, 50).replace('.wav', '')
        
        print(f"🎤 Клонирование голоса...")
        temp_ref_path = None
        temp_dir = temp_directory if temp_directory and os.path.exists(os.path.dirname(temp_directory)) else None
        
        try:
            if ref_audio is None:
                raise ValueError("ref_audio не может быть None")
            
            print("🔄 Подготовка референсного аудио (поддержка mp3, flac, wav, ogg)...")
            # Эта функция теперь гарантированно обработает любой формат
            audio_np, audio_sr = _prepare_audio_for_qwen(ref_audio, target_sr=24000)
            
            print(f"📊 Подготовленное аудио: {len(audio_np)/audio_sr:.2f} сек, {audio_sr}Hz")
            
            # Создаем временный файл с ПОДГОТОВЛЕННЫМИ данными
            temp_ref_path = _create_safe_temp_file(
                suffix='.wav', 
                data=(audio_np, audio_sr), # Передаем кортеж (data, sr)
                custom_dir=temp_dir
            )
            
            if os.path.exists(temp_ref_path):
                file_size = os.path.getsize(temp_ref_path)
                print(f"💾 Размер временного файла: {file_size/1024:.2f} KB")
                if file_size > 50 * 1024 * 1024:
                    raise ValueError("Слишком большой временный файл")
            
            gen_kwargs = {
                "text": text, "ref_audio": temp_ref_path, "ref_text": ref_text,
                "language": language if language != "Auto" else None,
                "x_vector_only_mode": (clone_mode == "x_vector_only"),
                "temperature": max(0.1, min(2.0, float(temperature))),
                "top_p": 0.9, "top_k": 50, "repetition_penalty": 1.05,
                "max_new_tokens": 1024, "do_sample": True,
            }
            
            start_time = time.time()
            wavs, sr = qwen_model.generate_voice_clone(**gen_kwargs)
            generation_time = time.time() - start_time
            
            audio_data = wavs[0] if isinstance(wavs, list) else wavs
            if torch.is_tensor(audio_data):
                audio_data = audio_data.detach().cpu().numpy()
            
            max_result_samples = 10 * 60 * sr
            if len(audio_data) > max_result_samples:
                audio_data = audio_data[:max_result_samples]
            
            audio_tensor = torch.from_numpy(audio_data).unsqueeze(0).unsqueeze(0)
            info = {
                "sample_rate": sr, "duration": round(len(audio_data) / sr, 2),
                "generation_time": round(generation_time, 2), "clone_mode": clone_mode,
                "temperature": temperature, "original_duration": round(len(audio_np) / audio_sr, 2),
                "original_sample_rate": audio_sr, "format_converted": True,
                "security": {"temp_file": "deleted" if temp_ref_path else "none", "input_sanitized": True}
            }
            print(f"✅ Клонирование успешно за {generation_time:.2f} сек")
            return ({"waveform": audio_tensor, "sample_rate": sr}, json.dumps(info, indent=2))
            
        except Exception as e:
            print(f"❌ Ошибка клонирования: {e}")
            traceback.print_exc()
            silence = torch.zeros((1, 1, 24000))
            return ({"waveform": silence, "sample_rate": 24000}, json.dumps({"error": str(e)[:200]}, indent=2))
        finally:
            if temp_ref_path and os.path.exists(temp_ref_path):
                try:
                    os.unlink(temp_ref_path)
                    print(f"🧹 Временный файл удален: {temp_ref_path}")
                except Exception as e:
                    print(f"⚠️ Не удалось удалить временный файл: {e}")

class QwenTTSBatchGenerate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "qwen_model": ("QWEN_TTS_MODEL",),
                "text_list": ("STRING", {"default": "Привет!|Как дела?|Пока!", "multiline": True}),
                "language": (["Auto", "Russian", "English", "Chinese", "German", "French", "Spanish", "Japanese", "Korean"], {"default": "Russian"}),
                "temperature": ("FLOAT", {"default": 0.8, "min": 0.1, "max": 2.0, "step": 0.1}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.1, "max": 1.0, "step": 0.05}),
                "separator": ("STRING", {"default": "|"}),
            },
            "optional": {
                "output_prefix": ("STRING", {"default": "batch_"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "speaker": ("STRING", {"default": "Vivian"}),
                "instruct": ("STRING", {"default": " ", "multiline": True}),
            }
        }
    RETURN_TYPES = ("AUDIO", "STRING", "STRING")
    RETURN_NAMES = ("audio", "info", "filenames")
    FUNCTION = "batch_generate"
    CATEGORY = "audio/tts"
    OUTPUT_IS_LIST = (True, False, False)

    def batch_generate(self, qwen_model, text_list, language, temperature, top_p, separator="|", output_prefix="batch_", seed=0, speaker="Vivian", instruct=" "):
        try: seed = int(seed) & 0xFFFFFFFF
        except (ValueError, TypeError): seed = 0
        if not separator or not isinstance(separator, str): separator = "|"
        separator = separator[:10]
        if not isinstance(text_list, str): text_list = str(text_list)
        texts = [t.strip() for t in text_list.split(separator) if t.strip()]
        max_batch_size = 20
        if len(texts) > max_batch_size:
            print(f"⚠️ Пакет обрезан с {len(texts)} до {max_batch_size} элементов")
            texts = texts[:max_batch_size]
        print(f"🔄 Пакетная генерация: {len(texts)} текстов")
        audio_outputs = []
        filenames = []
        successful = 0
        failed = 0
        safe_prefix = _sanitize_filename(output_prefix, 30).replace('.wav', '')
        
        for i, text in enumerate(texts):
            if not isinstance(text, str): text = str(text)
            text = text.strip()[:2000]
            print(f"  {i+1}/{len(texts)}: {text[:40]}...")
            try:
                item_seed = (seed + i * 7919) & 0xFFFFFFFF
                torch.manual_seed(item_seed)
                model_type = qwen_model.metadata.get("model_type", "base")
                start_time = time.time()
                safe_temp = max(0.1, min(2.0, float(temperature)))
                safe_top_p = max(0.1, min(1.0, float(top_p)))
                
                if model_type == "custom_voice":
                    safe_speaker = re.sub(r'[^\w\s\-]', '', str(speaker).strip())[:50] if speaker else "Vivian"
                    safe_instruct = str(instruct).strip()[:500] if instruct else ""
                    wavs, sr = qwen_model.generate_custom_voice(text=text, language=language if language != "Auto" else None, speaker=safe_speaker, instruct=safe_instruct, temperature=safe_temp, top_p=safe_top_p, max_new_tokens=1024)
                elif model_type == "voice_design":
                    safe_instruct = str(instruct).strip()[:500] if instruct else "естественный и чёткий голос"
                    wavs, sr = qwen_model.generate_voice_design(text=text, language=language if language != "Auto" else None, instruct=safe_instruct, temperature=safe_temp, top_p=safe_top_p, max_new_tokens=1024)
                else:
                    safe_instruct = str(instruct).strip()[:500] if instruct else "нейтральный тон"
                    wavs, sr = qwen_model.generate_voice_design(text=text, language=language if language != "Auto" else None, instruct=safe_instruct, temperature=safe_temp, top_p=safe_top_p, max_new_tokens=1024)
                
                duration = time.time() - start_time
                audio_data = wavs[0] if isinstance(wavs, list) else wavs
                if torch.is_tensor(audio_data): audio_data = audio_data.detach().cpu().numpy()
                max_samples = 5 * 60 * sr
                if len(audio_data) > max_samples: audio_data = audio_data[:max_samples]
                audio_tensor = torch.from_numpy(audio_data).unsqueeze(0).unsqueeze(0)
                filename = f"{safe_prefix}_{i+1:03d}.wav"
                filenames.append(filename)
                audio_outputs.append({"waveform": audio_tensor, "sample_rate": int(sr), "metadata": {"index": i, "text": text[:100], "duration": len(audio_data) / sr, "generation_time": duration}})
                successful += 1
                print(f"    ✅ Успешно за {duration:.2f} сек")
            except Exception as e:
                print(f"    ❌ Ошибка: {str(e)[:100]}")
                silence = torch.zeros((1, 1, 24000))
                audio_outputs.append({"waveform": silence, "sample_rate": 24000, "metadata": {"error": str(e)[:100], "index": i}})
                filenames.append(f"error_{i+1:03d}.wav")
                failed += 1
        
        audio_list = [{"waveform": item["waveform"], "sample_rate": item["sample_rate"]} for item in audio_outputs]
        info = {"total": len(texts), "successful": successful, "failed": failed, "batch_seed": seed, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
        filenames_str = separator.join(filenames)
        return (audio_list, json.dumps(info, indent=2), filenames_str)

class QwenTTSEmotionMixer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_inputs": ("AUDIO",),
                "weights": ("STRING", {"default": "1.0", "multiline": False}),
                "normalize": (["yes", "no"], {"default": "yes"}),
            }
        }
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "mix_emotions"
    CATEGORY = "audio/tts"
    INPUT_IS_LIST = True

    def mix_emotions(self, audio_inputs, weights, normalize):
        if not audio_inputs:
            print("⚠️ Нет входных аудио данных")
            silence = torch.zeros((1, 1, 24000))
            return ({"waveform": silence, "sample_rate": 24000},)
        print(f"🎚️ Микширование {len(audio_inputs)} аудио дорожек...")
        try:
            weight_str = weights[0] if isinstance(weights, list) and weights else "1.0"
            weight_parts = weight_str.split(',')
            weight_list = []
            for part in weight_parts:
                try:
                    weight = float(part.strip())
                    weight_list.append(max(0.0, min(10.0, weight)))
                except (ValueError, TypeError):
                    weight_list.append(1.0)
            while len(weight_list) < len(audio_inputs): weight_list.append(1.0)
            if len(weight_list) > len(audio_inputs): weight_list = weight_list[:len(audio_inputs)]
            
            waveforms = []
            sample_rates = []
            max_len = 0
            
            for i, audio in enumerate(audio_inputs):
                if audio is None: continue
                if isinstance(audio, dict) and "waveform" in audio:
                    wf = audio["waveform"]
                    sr = audio.get("sample_rate", 24000)
                elif torch.is_tensor(audio):
                    wf = audio
                    sr = 24000
                else:
                    print(f"⚠️ Неизвестный формат аудио #{i}")
                    continue
                if torch.is_tensor(wf): wf_np = wf.detach().cpu().numpy().squeeze()
                else: wf_np = np.array(wf).squeeze()
                if wf_np.size == 0:
                    print(f"⚠️ Пустые аудио данные #{i}")
                    continue
                waveforms.append(wf_np)
                sample_rates.append(sr)
                max_len = max(max_len, len(wf_np))
            
            if not waveforms:
                print("⚠️ Нет валидных аудио данных для микширования")
                silence = torch.zeros((1, 1, 24000))
                return ({"waveform": silence, "sample_rate": 24000},)
            
            target_sr = sample_rates[0]
            padded_waveforms = []
            for wf in waveforms:
                if len(wf) < max_len:
                    padded = np.zeros(max_len, dtype=wf.dtype)
                    padded[:len(wf)] = wf
                    padded_waveforms.append(padded)
                else:
                    padded_waveforms.append(wf[:max_len])
            
            norm_flag = normalize[0].lower() == "yes" if isinstance(normalize, list) else str(normalize).lower() == "yes"
            if norm_flag:
                weight_sum = sum(weight_list)
                if weight_sum > 0: weight_list = [w / weight_sum for w in weight_list]
                else: weight_list = [1.0 / len(weight_list)] * len(weight_list)
            
            mixed = np.zeros_like(padded_waveforms[0])
            for wf, weight in zip(padded_waveforms, weight_list):
                mixed += wf * weight
            
            max_val = np.max(np.abs(mixed))
            if max_val > 1.0: mixed = mixed / max_val
            
            mixed_tensor = torch.from_numpy(mixed).unsqueeze(0).unsqueeze(0)
            print(f"✅ Смешано {len(waveforms)} аудио дорожек")
            return ({"waveform": mixed_tensor, "sample_rate": target_sr},)
        except Exception as e:
            print(f"❌ Ошибка микширования: {e}")
            traceback.print_exc()
            silence = torch.zeros((1, 1, 24000))
            return ({"waveform": silence, "sample_rate": 24000},)

class QwenTTSAudioSaver:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "filename": ("STRING", {"default": "output.wav"}),
                "sample_rate": ("INT", {"default": 24000, "min": 8000, "max": 48000}),
            },
            "optional": {
                "output_dir": ("STRING", {"default": "output/tts", "multiline": False}),
                "metadata": ("STRING", {"default": " ", "multiline": True}),
            }
        }
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("filepath",)
    FUNCTION = "save_audio"
    CATEGORY = "audio/tts"

    def save_audio(self, audio, filename, sample_rate, output_dir="output/tts", metadata=" "):
        try:
            sample_rate = int(sample_rate)
            sample_rate = max(8000, min(48000, sample_rate))
        except (ValueError, TypeError): sample_rate = 24000
        
        if not output_dir or not isinstance(output_dir, str): output_dir = "output/tts"
        safe_output_dir = output_dir.strip()
        safe_output_dir = re.sub(r'\.\./', '', safe_output_dir)
        safe_output_dir = re.sub(r'\.\.\\', ' ', safe_output_dir)
        safe_output_dir = os.path.normpath(safe_output_dir)
        parts = safe_output_dir.split(os.sep)
        safe_parts = [] 
        for part in parts:
            if part and part not in ['.', '..']:
                clean_part = re.sub(r'[^\w\s\.\-]', '_', part)
                safe_parts.append(clean_part[:50])
        if not safe_parts: safe_parts = ["output", "tts"]
        safe_output_dir = os.sep.join(safe_parts)
        
        try: base_output_dir = folder_paths.get_output_directory()
        except Exception: base_output_dir = os.path.join(os.path.dirname(__file__), "output")
        
        full_output_dir = os.path.join(base_output_dir, safe_output_dir)
        full_output_dir = os.path.normpath(full_output_dir)
        if not os.path.normpath(full_output_dir).startswith(os.path.normpath(base_output_dir)):
            print(f"⚠️ Недопустимый путь, используется default")
            full_output_dir = os.path.join(base_output_dir, "output", "tts")
        
        try: os.makedirs(full_output_dir, exist_ok=True)
        except Exception as e:
            print(f"⚠️ Не удалось создать директорию: {e}")
            full_output_dir = base_output_dir
        
        safe_filename = _sanitize_filename(filename)
        filepath = os.path.join(full_output_dir, safe_filename)
        filepath = os.path.normpath(filepath)
        if not filepath.startswith(os.path.normpath(full_output_dir)):
            raise ValueError(f"Недопустимое имя файла (path traversal): {filename}")
        
        try:
            if isinstance(audio, dict) and "waveform" in audio:
                audio_np = audio["waveform"].detach().cpu().numpy().squeeze()
            elif torch.is_tensor(audio):
                audio_np = audio.detach().cpu().numpy().squeeze()
            else:
                audio_np = np.array(audio).squeeze()
        except Exception as e:
            print(f"❌ Ошибка обработки аудио: {e}")
            audio_np = np.zeros(sample_rate)
        
        max_duration = 30 * 60
        max_samples = max_duration * sample_rate
        if len(audio_np) > max_samples:
            audio_np = audio_np[:max_samples]
            print(f"⚠️ Аудио обрезано до {max_duration} минут")
        
        try:
            sf.write(filepath, audio_np, sample_rate, format='WAV', subtype='PCM_16')
            print(f"💾 Аудио сохранено: {filepath}")
        except Exception as e:
            print(f"❌ Ошибка сохранения аудио: {e}")
            alt_filename = f"output_{int(time.time())}.wav"
            alt_filepath = os.path.join(base_output_dir, alt_filename)
            try:
                sf.write(alt_filepath, audio_np, sample_rate, format='WAV', subtype='PCM_16')
                filepath = alt_filepath
                print(f"💾 Аудио сохранено в альтернативный путь: {filepath}")
            except Exception as e2:
                print(f"❌ Критическая ошибка сохранения: {e2}")
                raise
        
        if metadata and isinstance(metadata, str) and metadata.strip():
            safe_metadata = _safe_json_loads(metadata)
            if safe_metadata:
                try:
                    metadata_file = os.path.splitext(filepath)[0] + '.json'
                    safe_metadata.update({
                        "save_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "filename": safe_filename, "sample_rate": sample_rate,
                        "duration": len(audio_np) / sample_rate if sample_rate > 0 else 0,
                        "file_size": os.path.getsize(filepath) if os.path.exists(filepath) else 0,
                        "security": {"filename_sanitized": True, "path_validated": True}
                    })
                    with open(metadata_file, 'w', encoding='utf-8') as f:
                        json.dump(safe_metadata, f, indent=2, ensure_ascii=False)
                    print(f"📝 Метаданные сохранены: {metadata_file}")
                except Exception as e:
                    print(f"⚠️ Не удалось сохранить метаданные: {e}")
        
        return (filepath,)

NODE_CLASS_MAPPINGS = {
    "QwenTTSModelLoader": QwenTTSModelLoader,
    "QwenTTSGenerate": QwenTTSGenerate,
    "QwenTTSVoiceClone": QwenTTSVoiceClone,
    "QwenTTSBatchGenerate": QwenTTSBatchGenerate,
    "QwenTTSAudioSaver": QwenTTSAudioSaver,
    "QwenTTSEmotionMixer": QwenTTSEmotionMixer,
    "QwenTTSConfig": QwenTTSConfig,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenTTSModelLoader": "DVA 🤖 Qwen TTS Loader",
    "QwenTTSGenerate": "DVA 🎤 Qwen TTS Generate",
    "QwenTTSVoiceClone": "DVA 🎭 Qwen TTS Voice Clone",
    "QwenTTSBatchGenerate": "DVA 📚 Qwen TTS Batch Generate",
    "QwenTTSAudioSaver": "DVA 💾 Qwen TTS Audio Saver",
    "QwenTTSEmotionMixer": "DVA 🔀 Qwen TTS Emotion Mixer",
    "QwenTTSConfig": "DVA ⚙️ Qwen TTS Config",
}
