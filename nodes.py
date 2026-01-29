import os
import torch
import numpy as np
import folder_paths
import json
import time
import traceback
import tempfile
import soundfile as sf
import re
import uuid
import pathlib
import posixpath
import hashlib
from typing import List, Dict, Any

# Попробуем импортировать qwen_tts
try:
    from qwen_tts import Qwen3TTSModel
    QWEN_AVAILABLE = True
except ImportError:
    QWEN_AVAILABLE = False
    print("⚠️ Qwen-TTS не установлен. Установите: pip install qwen-tts")


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
    
    # Удаляем все пути и оставляем только имя файла
    base_name = os.path.basename(str(filename))
    
    # Заменяем опасные символы, разрешаем буквы, цифры, пробелы, точку, дефис, подчеркивание
    sanitized = re.sub(r'[^\w\s\.\-]', '_', base_name)
    
    # Убираем множественные подчеркивания
    sanitized = re.sub(r'_+', '_', sanitized)
    
    # Убираем начальные/конечные точки и пробелы
    sanitized = sanitized.strip('. ')
    
    # Обрезаем до максимальной длины
    if len(sanitized) > max_length:
        name, ext = os.path.splitext(sanitized)
        sanitized = name[:max_length - len(ext)] + ext
    
    # Убеждаемся, что не пустая строка
    if not sanitized or sanitized in ['.', '..']:
        sanitized = "output"
    
    # Добавляем расширение .wav если нет
    if not sanitized.lower().endswith('.wav'):
        sanitized += '.wav'
    
    return sanitized


def _safe_json_loads(json_str, max_size=10000):
    """Безопасная загрузка JSON с ограничениями"""
    if not json_str or not isinstance(json_str, str):
        return {}
    
    # Ограничиваем размер входных данных
    if len(json_str) > max_size:
        json_str = json_str[:max_size]
    
    try:
        parsed = json.loads(json_str)
        
        # Проверяем тип данных
        if not isinstance(parsed, (dict, list)):
            return {"data": str(parsed)[:500]}
        
        # Рекурсивная функция для ограничения глубины и размера
        def limit_structure(obj, depth=0, max_depth=10, max_items=100):
            if depth > max_depth:
                return "[max depth reached]"
            
            if isinstance(obj, dict):
                # Ограничиваем количество ключей
                items = list(obj.items())[:max_items]
                return {str(k)[:100]: limit_structure(v, depth+1, max_depth, max_items) 
                       for k, v in items}
            elif isinstance(obj, list):
                # Ограничиваем длину списка
                return [limit_structure(item, depth+1, max_depth, max_items) 
                       for item in obj[:max_items]]
            elif isinstance(obj, str):
                # Ограничиваем длину строк
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
    
    # Нормализуем путь
    normalized = os.path.normpath(path)
    
    # Проверяем на path traversal
    if '..' in normalized.split(os.sep):
        return False, "Путь содержит '..' (path traversal)"
    
    # Проверяем абсолютные пути
    if os.path.isabs(path) and allowed_base:
        # Проверяем, что путь находится внутри разрешенной базовой директории
        try:
            common = os.path.commonpath([normalized, allowed_base])
            if common != allowed_base:
                return False, "Путь находится вне разрешенной директории"
        except ValueError:
            return False, "Недопустимый путь"
    
    # Проверяем длину
    if len(path) > 500:
        return False, "Слишком длинный путь"
    
    return True, normalized


def _create_safe_temp_file(suffix='.wav', data=None, sample_rate=24000):
    """Создание безопасного временного файла"""
    safe_temp_dir = tempfile.gettempdir()
    temp_filename = f"qwen_tts_{uuid.uuid4().hex}_{int(time.time())}{suffix}"
    temp_path = os.path.join(safe_temp_dir, temp_filename)
    
    # Записываем данные если они предоставлены
    if data is not None:
        try:
            sf.write(temp_path, data, sample_rate)
        except Exception as e:
            raise ValueError(f"Не удалось записать временный файл: {e}")
    
    return temp_path


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
    
    def load_model(self, model_name, precision, attention_type, device, cache_dir=""):
        if not QWEN_AVAILABLE:
            raise ImportError("Qwen-TTS не установлен. Установите: pip install qwen-tts")
        
        print(f"🔄 Загрузка модели TTS: {model_name}")
        
        # Валидация входных данных
        if not isinstance(model_name, str):
            raise ValueError("model_name должен быть строкой")
        
        dtype_map = {
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "fp32": torch.float32,
        }
        dtype = dtype_map.get(precision, torch.float16)
        
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        try:
            kwargs = {
                "torch_dtype": dtype,
                "device_map": device,
                "low_cpu_mem_usage": True,
                "trust_remote_code": False,  # Безопасность по умолчанию
            }
            
            # Доверенные модели (из официального источника)
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
            else:
                print(f"⚠️ Модель {model_name} не в доверенном списке, remote_code отключен")
            
            # Безопасная проверка cache_dir
            if cache_dir and isinstance(cache_dir, str) and cache_dir.strip():
                cache_dir = cache_dir.strip()
                is_valid, validated_path = _validate_path(cache_dir)
                if is_valid and os.path.exists(os.path.dirname(validated_path)):
                    kwargs["cache_dir"] = validated_path
                else:
                    print(f"⚠️ Недопустимый cache_dir, используется значение по умолчанию")
            
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
                "text": ("STRING", {
                    "default": "Привет! Это тест синтеза речи.",
                    "multiline": True
                }),
                "language": (["Auto", "Russian", "English", "Chinese", "German", 
                            "French", "Spanish", "Japanese", "Korean"], {"default": "Russian"}),
                "temperature": ("FLOAT", {"default": 0.9, "min": 0.1, "max": 2.0, "step": 0.1}),
                "top_p": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 1.0, "step": 0.05}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "speaker": ("STRING", {"default": "Vivian"}),  # для CustomVoice
                "instruct": ("STRING", {"default": "", "multiline": True}),  # для CustomVoice/VoiceDesign
                "emotion_preset": (["neutral", "happy", "sad", "angry", "surprised", 
                                  "energetic", "calm", "dramatic", "professional"], 
                                 {"default": "neutral"}),
            }
        }
    
    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "info")
    FUNCTION = "generate_speech"
    CATEGORY = "audio/tts"
    
    def generate_speech(self, qwen_model, text, language, temperature, top_p, seed,
                       speaker="Vivian", instruct="", emotion_preset="neutral"):
        
        # Безопасная обработка seed
        try:
            seed = int(seed) & 0xFFFFFFFF  # Ограничиваем размер seed
        except (ValueError, TypeError):
            seed = 0
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        # Безопасная обработка текста
        if not isinstance(text, str):
            text = str(text)
        text = text.strip()[:5000]  # Ограничиваем длину текста
        
        # Безопасная обработка speaker
        if speaker and isinstance(speaker, str):
            speaker = re.sub(r'[^\w\s\-]', '', speaker.strip())[:50]
        
        # Безопасная обработка instruct
        if instruct and isinstance(instruct, str):
            instruct = instruct.strip()[:1000]
        
        # Эмоции → instruct
        emotion_to_instruct = {
            "neutral": "",
            "happy": "радостный и энергичный тон",
            "sad": "грустный и медленный тон",
            "angry": "сердитый и резкий тон",
            "surprised": "удивлённый, с высокой интонацией",
            "energetic": "быстрый и энергичный темп",
            "calm": "спокойный и мягкий тон",
            "dramatic": "театральный и выразительный тон",
            "professional": "чёткий и деловой тон",
        }
        
        if emotion_preset != "neutral" and not instruct:
            instruct = emotion_to_instruct.get(emotion_preset, "")
        
        model_type = qwen_model.metadata.get("model_type", "base")
        lang = None if language == "Auto" else language
        
        print(f"🎤 Генерация: {text[:50]}...")
        print(f"⚙️ Тип модели: {model_type}, язык: {language}, эмоция: {emotion_preset}")
        
        try:
            start_time = time.time()
            
            # Безопасная генерация в зависимости от типа модели
            safe_temperature = max(0.1, min(2.0, float(temperature)))
            safe_top_p = max(0.1, min(1.0, float(top_p)))
            
            if model_type == "custom_voice":
                if not speaker:
                    speaker = "Vivian"
                
                wavs, sr = qwen_model.generate_custom_voice(
                    text=text,
                    language=lang,
                    speaker=speaker,
                    instruct=instruct,
                    temperature=safe_temperature,
                    top_p=safe_top_p,
                    max_new_tokens=1024,
                )
            elif model_type == "voice_design":
                if not instruct:
                    instruct = "естественный и чёткий голос"
                
                wavs, sr = qwen_model.generate_voice_design(
                    text=text,
                    language=lang,
                    instruct=instruct,
                    temperature=safe_temperature,
                    top_p=safe_top_p,
                    max_new_tokens=1024,
                )
            else:  # base модели
                # Для base моделей нужен хотя бы минимальный текст инструкции
                if not instruct:
                    instruct = "нейтральный тон"
                
                # Генерация без референсного аудио
                wavs, sr = qwen_model.generate_voice_design(
                    text=text,
                    language=lang,
                    instruct=instruct,
                    temperature=safe_temperature,
                    top_p=safe_top_p,
                    max_new_tokens=1024,
                )
            
            duration = time.time() - start_time
            
            # Безопасная обработка аудиоданных
            audio_data = wavs[0] if isinstance(wavs, list) else wavs
            if torch.is_tensor(audio_data):
                audio_data = audio_data.detach().cpu().numpy()
            
            # Проверяем размер аудио (не более 10 минут)
            max_samples = 10 * 60 * sr  # 10 минут
            if len(audio_data) > max_samples:
                audio_data = audio_data[:max_samples]
                print(f"⚠️ Аудио обрезано до 10 минут")
            
            # Формат ComfyUI: [B, C, T] → [1, 1, T]
            audio_tensor = torch.from_numpy(audio_data).unsqueeze(0).unsqueeze(0)
            
            info = {
                "sample_rate": int(sr),
                "duration_sec": round(len(audio_data) / sr, 2),
                "generation_time_sec": round(duration, 2),
                "model": qwen_model.metadata.get("model_name", "unknown"),
                "model_type": model_type,
                "parameters": {
                    "language": language,
                    "speaker": speaker if model_type == "custom_voice" else "default",
                    "instruct": instruct[:100] if instruct else "",
                    "emotion_preset": emotion_preset,
                    "text_length": len(text),
                },
                "security": {
                    "trusted_model": qwen_model.metadata.get("trusted", False),
                    "input_sanitized": True,
                }
            }
            
            print(f"✅ Готово за {duration:.2f} сек, длина: {info['duration_sec']} сек")
            return ({"waveform": audio_tensor, "sample_rate": int(sr)}, json.dumps(info, indent=2, ensure_ascii=False))
            
        except Exception as e:
            print(f"❌ Ошибка генерации: {e}")
            traceback.print_exc()
            silence = torch.zeros((1, 1, 24000))
            sr = 24000
            return ({"waveform": silence, "sample_rate": sr}, json.dumps({"error": str(e)[:200]}, indent=2))


class QwenTTSVoiceClone:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "qwen_model": ("QWEN_TTS_MODEL",),
                "text": ("STRING", {
                    "default": "Привет! Это клонированный голос.",
                    "multiline": True
                }),
                "ref_audio": ("AUDIO",),
                "ref_text": ("STRING", {
                    "default": "Это текст из референсного аудио.",
                    "multiline": True
                }),
                "language": (["Russian", "English", "Chinese", "Auto"], {"default": "Russian"}),
                "clone_mode": (["x_vector_only", "icl"], {"default": "x_vector_only"}),
                "temperature": ("FLOAT", {"default": 0.8, "min": 0.1, "max": 2.0, "step": 0.1}),
            },
            "optional": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "output_prefix": ("STRING", {"default": "clone_"}),
            }
        }
    
    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "info")
    FUNCTION = "clone_voice"
    CATEGORY = "audio/tts"
    
    def clone_voice(self, qwen_model, text, ref_audio, ref_text, language, 
                   clone_mode, temperature, seed=0, output_prefix="clone_"):
        
        model_type = qwen_model.metadata.get("model_type", "base")
        if model_type != "base":
            raise ValueError("Клонирование работает только с моделями типа '-Base'")
        
        # Безопасная обработка seed
        try:
            seed = int(seed) & 0xFFFFFFFF
        except (ValueError, TypeError):
            seed = 0
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        # Безопасная обработка текста
        if not isinstance(text, str):
            text = str(text)
        text = text.strip()[:5000]
        
        if not isinstance(ref_text, str):
            ref_text = str(ref_text)
        ref_text = ref_text.strip()[:5000]
        
        # Безопасная обработка output_prefix
        if output_prefix and isinstance(output_prefix, str):
            output_prefix = _sanitize_filename(output_prefix, 50).replace('.wav', '')
        
        print(f"🎤 Клонирование голоса...")
        
        # Безопасный путь для временного файла
        temp_ref_path = None
        
        try:
            # === Безопасная обработка референсного аудио ===
            if ref_audio is None:
                raise ValueError("ref_audio не может быть None")
            
            # Извлекаем аудиоданные
            if isinstance(ref_audio, dict) and "waveform" in ref_audio:
                ref_wave = ref_audio["waveform"]
                ref_sr = ref_audio.get("sample_rate", 24000)
            elif torch.is_tensor(ref_audio):
                ref_wave = ref_audio
                ref_sr = 24000
            else:
                ref_wave = np.array(ref_audio, dtype=np.float32)
                ref_sr = 24000
            
            # Конвертируем в numpy
            if torch.is_tensor(ref_wave):
                ref_np = ref_wave.detach().cpu().numpy()
            else:
                ref_np = np.array(ref_wave, dtype=np.float32)
            
            # Приводим к форме [T]
            ref_np = ref_np.squeeze()
            
            # Проверяем размер (не более 30 секунд для клонирования)
            max_clone_samples = 30 * ref_sr
            if len(ref_np) > max_clone_samples:
                ref_np = ref_np[:max_clone_samples]
                print(f"⚠️ Референсное аудио обрезано до 30 секунд")
            
            # Нормализация диапазона
            if ref_np.dtype != np.float32:
                ref_np = ref_np.astype(np.float32)
            
            # Проверяем диапазон значений
            max_val = np.max(np.abs(ref_np))
            if max_val > 0:
                ref_np = np.clip(ref_np / max_val, -1.0, 1.0)
            
            # Создаем безопасный временный файл
            temp_ref_path = _create_safe_temp_file(suffix='.wav', data=ref_np, sample_rate=ref_sr)
            
            # Проверяем размер файла (не более 50MB)
            file_size = os.path.getsize(temp_ref_path)
            if file_size > 50 * 1024 * 1024:  # 50 MB
                raise ValueError("Слишком большой временный файл")
            
            gen_kwargs = {
                "text": text,
                "ref_audio": temp_ref_path,
                "ref_text": ref_text,
                "language": language if language != "Auto" else None,
                "x_vector_only_mode": (clone_mode == "x_vector_only"),
                "temperature": max(0.1, min(2.0, float(temperature))),
                "top_p": 0.9,
                "top_k": 50,
                "repetition_penalty": 1.05,
                "max_new_tokens": 1024,
                "do_sample": True,
            }
            
            start_time = time.time()
            wavs, sr = qwen_model.generate_voice_clone(**gen_kwargs)
            generation_time = time.time() - start_time
            
            # Обработка результата
            audio_data = wavs[0] if isinstance(wavs, list) else wavs
            if torch.is_tensor(audio_data):
                audio_data = audio_data.detach().cpu().numpy()
            
            # Проверяем размер результата
            max_result_samples = 10 * 60 * sr  # 10 минут
            if len(audio_data) > max_result_samples:
                audio_data = audio_data[:max_result_samples]
                print(f"⚠️ Результат клонирования обрезан до 10 минут")
            
            audio_tensor = torch.from_numpy(audio_data).unsqueeze(0).unsqueeze(0)
            
            info = {
                "sample_rate": sr,
                "duration": round(len(audio_data) / sr, 2),
                "generation_time": round(generation_time, 2),
                "clone_mode": clone_mode,
                "temperature": temperature,
                "original_duration": round(len(ref_np) / ref_sr, 2),
                "security": {
                    "temp_file": "deleted",
                    "input_sanitized": True,
                }
            }
            
            print(f"✅ Клонирование успешно за {generation_time:.2f} сек")
            
            return ({"waveform": audio_tensor, "sample_rate": sr}, json.dumps(info, indent=2))
            
        except Exception as e:
            print(f"❌ Ошибка клонирования: {e}")
            traceback.print_exc()
            silence = torch.zeros((1, 1, 24000))
            sr = 24000
            return ({"waveform": silence, "sample_rate": sr}, json.dumps({"error": str(e)[:200]}, indent=2))
        
        finally:
            # ГАРАНТИРОВАННОЕ удаление временного файла
            if temp_ref_path and os.path.exists(temp_ref_path):
                try:
                    os.unlink(temp_ref_path)
                except Exception as e:
                    print(f"⚠️ Не удалось удалить временный файл: {e}")


class QwenTTSBatchGenerate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "qwen_model": ("QWEN_TTS_MODEL",),
                "text_list": ("STRING", {
                    "default": "Привет!|Как дела?|Пока!",
                    "multiline": True
                }),
                "language": (["Auto", "Russian", "English", "Chinese", "German", 
                            "French", "Spanish", "Japanese", "Korean"], {"default": "Russian"}),
                "temperature": ("FLOAT", {"default": 0.8, "min": 0.1, "max": 2.0, "step": 0.1}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.1, "max": 1.0, "step": 0.05}),
                "separator": ("STRING", {"default": "|"}),
            },
            "optional": {
                "output_prefix": ("STRING", {"default": "batch_"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "speaker": ("STRING", {"default": "Vivian"}),
                "instruct": ("STRING", {"default": "", "multiline": True}),
            }
        }
    
    RETURN_TYPES = ("AUDIO", "STRING", "STRING")
    RETURN_NAMES = ("audio", "info", "filenames")
    FUNCTION = "batch_generate"
    CATEGORY = "audio/tts"
    OUTPUT_IS_LIST = (True, False, False)
    
    def batch_generate(self, qwen_model, text_list, language, temperature, top_p, separator="|",
                      output_prefix="batch_", seed=0, speaker="Vivian", instruct=""):
        
        # Безопасная обработка seed
        try:
            seed = int(seed) & 0xFFFFFFFF
        except (ValueError, TypeError):
            seed = 0
        
        # Безопасная обработка separator
        if not separator or not isinstance(separator, str):
            separator = "|"
        separator = separator[:10]  # Ограничиваем длину разделителя
        
        # Безопасная обработка text_list
        if not isinstance(text_list, str):
            text_list = str(text_list)
        
        # Разделяем текст с ограничением количества элементов
        texts = [t.strip() for t in text_list.split(separator) if t.strip()]
        
        # Ограничиваем количество элементов в пакете
        max_batch_size = 20
        if len(texts) > max_batch_size:
            print(f"⚠️ Пакет обрезан с {len(texts)} до {max_batch_size} элементов")
            texts = texts[:max_batch_size]
        
        print(f"🔄 Пакетная генерация: {len(texts)} текстов")
        
        audio_outputs = []
        filenames = []
        successful = 0
        failed = 0
        
        # Безопасная обработка output_prefix
        safe_prefix = _sanitize_filename(output_prefix, 30).replace('.wav', '')
        
        for i, text in enumerate(texts):
            if not isinstance(text, str):
                text = str(text)
            
            text = text.strip()[:2000]  # Дополнительное ограничение для пакетного режима
            
            print(f"  {i+1}/{len(texts)}: {text[:40]}...")
            
            try:
                # Используем уникальный seed для каждого элемента
                item_seed = (seed + i * 7919) & 0xFFFFFFFF  # Простое число для разнообразия
                torch.manual_seed(item_seed)
                
                # Определяем тип модели
                model_type = qwen_model.metadata.get("model_type", "base")
                
                start_time = time.time()
                
                # Безопасные параметры
                safe_temp = max(0.1, min(2.0, float(temperature)))
                safe_top_p = max(0.1, min(1.0, float(top_p)))
                
                # Генерация в зависимости от типа модели
                if model_type == "custom_voice":
                    safe_speaker = re.sub(r'[^\w\s\-]', '', str(speaker).strip())[:50] if speaker else "Vivian"
                    safe_instruct = str(instruct).strip()[:500] if instruct else ""
                    
                    wavs, sr = qwen_model.generate_custom_voice(
                        text=text,
                        language=language if language != "Auto" else None,
                        speaker=safe_speaker,
                        instruct=safe_instruct,
                        temperature=safe_temp,
                        top_p=safe_top_p,
                        max_new_tokens=1024,
                    )
                elif model_type == "voice_design":
                    safe_instruct = str(instruct).strip()[:500] if instruct else "естественный и чёткий голос"
                    
                    wavs, sr = qwen_model.generate_voice_design(
                        text=text,
                        language=language if language != "Auto" else None,
                        instruct=safe_instruct,
                        temperature=safe_temp,
                        top_p=safe_top_p,
                        max_new_tokens=1024,
                    )
                else:  # base модели
                    safe_instruct = str(instruct).strip()[:500] if instruct else "нейтральный тон"
                    
                    wavs, sr = qwen_model.generate_voice_design(
                        text=text,
                        language=language if language != "Auto" else None,
                        instruct=safe_instruct,
                        temperature=safe_temp,
                        top_p=safe_top_p,
                        max_new_tokens=1024,
                    )
                
                duration = time.time() - start_time
                
                # Обработка аудио
                audio_data = wavs[0] if isinstance(wavs, list) else wavs
                if torch.is_tensor(audio_data):
                    audio_data = audio_data.detach().cpu().numpy()
                
                # Ограничение длины
                max_samples = 5 * 60 * sr  # 5 минут для пакетного режима
                if len(audio_data) > max_samples:
                    audio_data = audio_data[:max_samples]
                
                audio_tensor = torch.from_numpy(audio_data).unsqueeze(0).unsqueeze(0)
                
                # Создаем имя файла
                filename = f"{safe_prefix}_{i+1:03d}.wav"
                filenames.append(filename)
                
                audio_outputs.append({
                    "waveform": audio_tensor,
                    "sample_rate": int(sr),
                    "metadata": {
                        "index": i,
                        "text": text[:100],
                        "duration": len(audio_data) / sr,
                        "generation_time": duration,
                    }
                })
                
                successful += 1
                print(f"    ✅ Успешно за {duration:.2f} сек")
                
            except Exception as e:
                print(f"    ❌ Ошибка: {str(e)[:100]}")
                # Создаем тишину в случае ошибки
                silence = torch.zeros((1, 1, 24000))
                audio_outputs.append({
                    "waveform": silence,
                    "sample_rate": 24000,
                    "metadata": {"error": str(e)[:100], "index": i}
                })
                filenames.append(f"error_{i+1:03d}.wav")
                failed += 1
        
        # Формируем выходные данные
        audio_list = [{"waveform": item["waveform"], "sample_rate": item["sample_rate"]} 
                     for item in audio_outputs]
        
        info = {
            "total": len(texts),
            "successful": successful,
            "failed": failed,
            "batch_seed": seed,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        
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
            # Безопасная обработка весов
            weight_str = weights[0] if isinstance(weights, list) and weights else "1.0"
            weight_parts = weight_str.split(',')
            
            weight_list = []
            for part in weight_parts:
                try:
                    weight = float(part.strip())
                    weight_list.append(max(0.0, min(10.0, weight)))  # Ограничиваем диапазон
                except (ValueError, TypeError):
                    weight_list.append(1.0)
            
            # Если весов меньше чем аудио, дополняем
            while len(weight_list) < len(audio_inputs):
                weight_list.append(1.0)
            
            # Если весов больше, обрезаем
            if len(weight_list) > len(audio_inputs):
                weight_list = weight_list[:len(audio_inputs)]
            
            # Извлекаем и обрабатываем аудиоданные
            waveforms = []
            sample_rates = []
            max_len = 0
            
            for i, audio in enumerate(audio_inputs):
                if audio is None:
                    continue
                    
                # Извлекаем waveform
                if isinstance(audio, dict) and "waveform" in audio:
                    wf = audio["waveform"]
                    sr = audio.get("sample_rate", 24000)
                elif torch.is_tensor(audio):
                    wf = audio
                    sr = 24000
                else:
                    print(f"⚠️ Неизвестный формат аудио #{i}")
                    continue
                
                # Конвертируем в numpy
                if torch.is_tensor(wf):
                    wf_np = wf.detach().cpu().numpy().squeeze()
                else:
                    wf_np = np.array(wf).squeeze()
                
                # Проверяем данные
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
            
            # Нормализуем sample rates (используем первый валидный)
            target_sr = sample_rates[0]
            for i, sr in enumerate(sample_rates):
                if sr != target_sr:
                    print(f"⚠️ Разные sample rates, возможны артефакты: {sr} vs {target_sr}")
            
            # Приводим все к одной длине
            padded_waveforms = []
            for wf in waveforms:
                if len(wf) < max_len:
                    # Дополняем тишиной
                    padded = np.zeros(max_len, dtype=wf.dtype)
                    padded[:len(wf)] = wf
                    padded_waveforms.append(padded)
                else:
                    padded_waveforms.append(wf[:max_len])
            
            # Нормализуем веса если нужно
            if normalize[0].lower() == "yes" if isinstance(normalize, list) else str(normalize).lower() == "yes":
                weight_sum = sum(weight_list)
                if weight_sum > 0:
                    weight_list = [w / weight_sum for w in weight_list]
                else:
                    weight_list = [1.0 / len(weight_list)] * len(weight_list)
            
            # Микшируем
            mixed = np.zeros_like(padded_waveforms[0])
            for wf, weight in zip(padded_waveforms, weight_list):
                mixed += wf * weight
            
            # Нормализуем результат чтобы избежать клиппинга
            max_val = np.max(np.abs(mixed))
            if max_val > 1.0:
                mixed = mixed / max_val
            
            # Конвертируем обратно в тензор
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
                "metadata": ("STRING", {"default": "", "multiline": True}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("filepath",)
    FUNCTION = "save_audio"
    CATEGORY = "audio/tts"
    
    def save_audio(self, audio, filename, sample_rate, output_dir="output/tts", metadata=""):
        # Валидация sample_rate
        try:
            sample_rate = int(sample_rate)
            sample_rate = max(8000, min(48000, sample_rate))
        except (ValueError, TypeError):
            sample_rate = 24000
        
        # Безопасная обработка output_dir
        if not output_dir or not isinstance(output_dir, str):
            output_dir = "output/tts"
        
        # Защита от path traversal
        safe_output_dir = output_dir.strip()
        
        # Удаляем все попытки уйти вверх по директории
        safe_output_dir = re.sub(r'\.\./', '', safe_output_dir)
        safe_output_dir = re.sub(r'\.\.\\', '', safe_output_dir)
        
        # Нормализуем путь
        safe_output_dir = os.path.normpath(safe_output_dir)
        
        # Проверяем каждый компонент пути
        parts = safe_output_dir.split(os.sep)
        safe_parts = []
        for part in parts:
            if part and part not in ['.', '..']:
                # Очищаем и ограничиваем длину
                clean_part = re.sub(r'[^\w\s\.\-]', '_', part)
                safe_parts.append(clean_part[:50])
        
        # Если после очистки путь пустой, используем дефолтный
        if not safe_parts:
            safe_parts = ["output", "tts"]
        
        safe_output_dir = os.sep.join(safe_parts)
        
        # Получаем базовую директорию вывода ComfyUI
        try:
            base_output_dir = folder_paths.get_output_directory()
        except Exception:
            base_output_dir = os.path.join(os.path.dirname(__file__), "output")
        
        # Создаем полный путь
        full_output_dir = os.path.join(base_output_dir, safe_output_dir)
        
        # Финальная проверка пути
        full_output_dir = os.path.normpath(full_output_dir)
        if not os.path.normpath(full_output_dir).startswith(os.path.normpath(base_output_dir)):
            print(f"⚠️ Недопустимый путь, используется default")
            full_output_dir = os.path.join(base_output_dir, "output", "tts")
        
        # Создаем директорию
        try:
            os.makedirs(full_output_dir, exist_ok=True)
        except Exception as e:
            print(f"⚠️ Не удалось создать директорию: {e}")
            full_output_dir = base_output_dir
        
        # Безопасное имя файла
        safe_filename = _sanitize_filename(filename)
        
        # Финальная проверка пути
        filepath = os.path.join(full_output_dir, safe_filename)
        filepath = os.path.normpath(filepath)
        
        # Проверяем, что файл остается внутри целевой директории
        if not filepath.startswith(os.path.normpath(full_output_dir)):
            raise ValueError(f"Недопустимое имя файла (path traversal): {filename}")
        
        # Безопасная обработка аудио
        try:
            if isinstance(audio, dict) and "waveform" in audio:
                audio_np = audio["waveform"].detach().cpu().numpy().squeeze()
            elif torch.is_tensor(audio):
                audio_np = audio.detach().cpu().numpy().squeeze()
            else:
                audio_np = np.array(audio).squeeze()
        except Exception as e:
            print(f"❌ Ошибка обработки аудио: {e}")
            audio_np = np.zeros(sample_rate)  # 1 секунда тишины
        
        # Проверяем размер аудио
        max_duration = 30 * 60  # 30 минут максимум
        max_samples = max_duration * sample_rate
        
        if len(audio_np) > max_samples:
            audio_np = audio_np[:max_samples]
            print(f"⚠️ Аудио обрезано до {max_duration} минут")
        
        # Сохраняем файл
        try:
            sf.write(filepath, audio_np, sample_rate)
            print(f"💾 Аудио сохранено: {filepath}")
        except Exception as e:
            print(f"❌ Ошибка сохранения аудио: {e}")
            # Пробуем альтернативный путь
            alt_filename = f"output_{int(time.time())}.wav"
            alt_filepath = os.path.join(base_output_dir, alt_filename)
            try:
                sf.write(alt_filepath, audio_np, sample_rate)
                filepath = alt_filepath
                print(f"💾 Аудио сохранено в альтернативный путь: {filepath}")
            except Exception as e2:
                print(f"❌ Критическая ошибка сохранения: {e2}")
                raise
        
        # Безопасная обработка метаданных
        if metadata and isinstance(metadata, str) and metadata.strip():
            safe_metadata = _safe_json_loads(metadata)
            
            if safe_metadata:
                try:
                    metadata_file = os.path.splitext(filepath)[0] + '.json'
                    
                    # Добавляем системную информацию
                    safe_metadata.update({
                        "save_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "filename": safe_filename,
                        "sample_rate": sample_rate,
                        "duration": len(audio_np) / sample_rate if sample_rate > 0 else 0,
                        "file_size": os.path.getsize(filepath) if os.path.exists(filepath) else 0,
                        "security": {
                            "filename_sanitized": True,
                            "path_validated": True,
                        }
                    })
                    
                    with open(metadata_file, 'w', encoding='utf-8') as f:
                        json.dump(safe_metadata, f, indent=2, ensure_ascii=False)
                    
                    print(f"📝 Метаданные сохранены: {metadata_file}")
                except Exception as e:
                    print(f"⚠️ Не удалось сохранить метаданные: {e}")
        
        return (filepath,)


# === РЕГИСТРАЦИЯ НОД ===
NODE_CLASS_MAPPINGS = {
    "QwenTTSModelLoader": QwenTTSModelLoader,
    "QwenTTSGenerate": QwenTTSGenerate,
    "QwenTTSVoiceClone": QwenTTSVoiceClone,
    "QwenTTSBatchGenerate": QwenTTSBatchGenerate,
    "QwenTTSAudioSaver": QwenTTSAudioSaver,
    "QwenTTSEmotionMixer": QwenTTSEmotionMixer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenTTSModelLoader": "DVA 🤖 Qwen TTS Loader",
    "QwenTTSGenerate": "DVA 🎤 Qwen TTS Generate",
    "QwenTTSVoiceClone": "DVA 🎭 Qwen TTS Voice Clone",
    "QwenTTSBatchGenerate": "DVA 📚 Qwen TTS Batch Generate",
    "QwenTTSAudioSaver": "DVA 💾 Qwen TTS Audio Saver",
    "QwenTTSEmotionMixer": "DVA 🔀 Qwen TTS Emotion Mixer",
}
