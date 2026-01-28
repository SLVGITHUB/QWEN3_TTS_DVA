import os
import shutil
import torch
import numpy as np
import folder_paths
import json
import time
import traceback
import tempfile
import soundfile as sf
import re
from typing import List, Tuple, Dict

# Попробуем импортировать qwen_tts
try:
    from qwen_tts import Qwen3TTSModel
    QWEN_AVAILABLE = True
except ImportError:
    QWEN_AVAILABLE = False
    print("⚠️ Qwen-TTS не установлен. Установите: pip install qwen-tts")


def _detect_model_type(model_name):
    if "CustomVoice" in model_name:
        return "custom_voice"
    elif "VoiceDesign" in model_name:
        return "voice_design"
    else:
        return "base"


class PathSanitizer:
    """Класс для безопасной обработки путей и имен файлов"""
    
    @staticmethod
    def sanitize_path(path: str, allow_absolute: bool = False) -> str:
        """
        Очищает путь от path traversal атак и опасных символов
        
        Args:
            path: Исходный путь
            allow_absolute: Разрешить абсолютные пути (только если они внутри разрешенных директорий)
        
        Returns:
            Очищенный безопасный путь
        """
        if not path or not isinstance(path, str):
            return ""
        
        # Удаляем все попытки path traversal
        cleaned = re.sub(r'\.\./|\.\.\\', '', path)
        
        # Заменяем множественные разделители
        cleaned = re.sub(r'[/\\]{2,}', '/', cleaned)
        
        # Убираем опасные символы (разрешаем буквы, цифры, пробел, -_./)
        safe_chars = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
                        '0123456789'
                        '_-./ \\')
        cleaned = ''.join(c for c in cleaned if c in safe_chars)
        
        # Удаляем ведущие/завершающие пробелы и точки
        cleaned = cleaned.strip(' .')
        
        # Если путь должен быть абсолютным, проверяем его безопасность
        if allow_absolute and os.path.isabs(cleaned):
            # Для абсолютных путей проверяем, что они внутри разрешенных директорий
            allowed_dirs = [
                folder_paths.get_output_directory(),
                folder_paths.get_temp_directory(),
                folder_paths.get_input_directory(),
                os.path.expanduser("~"),
            ]
            
            # Нормализуем путь
            cleaned = os.path.normpath(cleaned)
            
            # Проверяем, что путь начинается с одной из разрешенных директорий
            is_safe = False
            for allowed_dir in allowed_dirs:
                allowed_norm = os.path.normpath(allowed_dir)
                if cleaned.startswith(allowed_norm):
                    is_safe = True
                    break
            
            if not is_safe:
                # Если путь небезопасный, преобразуем в относительный
                cleaned = os.path.basename(cleaned)
        
        return cleaned
    
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """
        Очищает имя файла от опасных символов и path traversal
        
        Args:
            filename: Исходное имя файла
        
        Returns:
            Очищенное безопасное имя файла
        """
        if not filename or not isinstance(filename, str):
            return "output.wav"
        
        # Удаляем path traversal
        cleaned = re.sub(r'\.\./|\.\.\\', '', filename)
        
        # Берем только имя файла (не путь)
        basename = os.path.basename(cleaned)
        
        # Удаляем опасные символы
        safe_chars = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
                        '0123456789'
                        '_- .')
        cleaned = ''.join(c for c in basename if c in safe_chars)
        
        # Удаляем ведущие/завершающие пробелы и точки
        cleaned = cleaned.strip(' .')
        
        # Ограничиваем длину (максимум 255 символов)
        if len(cleaned) > 255:
            name, ext = os.path.splitext(cleaned)
            cleaned = name[:250] + ext
        
        # Если имя файла пустое после очистки
        if not cleaned:
            cleaned = "output.wav"
        
        return cleaned
    
    @staticmethod
    def ensure_safe_extension(filename: str, default_ext: str = ".wav") -> str:
        """
        Убеждается, что у файла безопасное расширение
        
        Args:
            filename: Имя файла
            default_ext: Расширение по умолчанию
        
        Returns:
            Имя файла с безопасным расширением
        """
        if not filename:
            return f"output{default_ext}"
        
        # Получаем расширение
        name, ext = os.path.splitext(filename)
        
        # Разрешенные аудио расширения
        allowed_extensions = {'.wav', '.mp3', '.ogg', '.flac', '.m4a'}
        
        # Если расширение не разрешено или отсутствует, используем default_ext
        if not ext or ext.lower() not in allowed_extensions:
            ext = default_ext
        
        # Очищаем имя файла
        safe_name = PathSanitizer.sanitize_filename(name)
        
        return safe_name + ext
    
    @staticmethod
    def create_secure_temp_file(suffix: str = ".wav") -> str:
        """
        Создает безопасный временный файл
        
        Args:
            suffix: Расширение файла
        
        Returns:
            Путь к временному файлу
        """
        # Используем tempfile с явными параметрами безопасности
        temp_dir = tempfile.gettempdir()
        safe_suffix = PathSanitizer.sanitize_filename(suffix)
        
        # Создаем временный файл с безопасным именем
        fd, temp_path = tempfile.mkstemp(suffix=safe_suffix, dir=temp_dir)
        os.close(fd)
        
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
        
        dtype_map = {
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "fp32": torch.float32,
        }
        dtype = dtype_map[precision]
        
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        try:
            kwargs = {
                "torch_dtype": dtype,
                "device_map": device,
                "low_cpu_mem_usage": True,
                "trust_remote_code": True,
            }
            
            # Безопасная обработка cache_dir
            if cache_dir and isinstance(cache_dir, str):
                safe_cache_dir = PathSanitizer.sanitize_path(cache_dir, allow_absolute=True)
                if safe_cache_dir and os.path.exists(safe_cache_dir):
                    kwargs["cache_dir"] = safe_cache_dir
                    print(f"📁 Используется cache_dir: {safe_cache_dir}")
            
            try:
                kwargs["attn_implementation"] = attention_type
                model = Qwen3TTSModel.from_pretrained(model_name, **kwargs)
            except Exception:
                print("⚠️ Fallback на eager attention")
                kwargs["attn_implementation"] = "eager"
                model = Qwen3TTSModel.from_pretrained(model_name, **kwargs)
            
            print(f"✅ Модель загружена на {model.device}")
            
            model.metadata = {
                "model_name": model_name,
                "precision": precision,
                "attention_type": attention_type,
                "device": str(model.device),
                "model_type": _detect_model_type(model_name),
            }
            
            return (model,)
            
        except Exception as e:
            print(f"❌ Ошибка загрузки модели: {e}")
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
        
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
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
        if instruct:
            print(f"📋 Инструкция: {instruct}")
        
        try:
            start_time = time.time()
            
            if model_type == "custom_voice":
                wavs, sr = qwen_model.generate_custom_voice(
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
                    instruct = "естественный и чёткий голос"
                wavs, sr = qwen_model.generate_voice_design(
                    text=text,
                    language=lang,
                    instruct=instruct,
                    temperature=temperature,
                    top_p=top_p,
                    max_new_tokens=1024,
                )
            else:  # base — без ref_audio использовать нельзя
                raise ValueError("Модель типа 'Base' требует референсного аудио. Используйте CustomVoice или VoiceDesign.")
            
            duration = time.time() - start_time
            audio_data = wavs[0] if isinstance(wavs, list) else wavs
            if torch.is_tensor(audio_data):
                audio_data = audio_data.cpu().numpy()
            
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
            return ({"waveform": silence, "sample_rate": sr}, f"Error: {str(e)}")


class EmotionControlParameters:
    """Класс для управления эмоциональными параметрами голоса"""
    
    def __init__(self):
        # Параметры по умолчанию (нейтральные)
        self.params = {
            "tempo": 1.0,           # темп речи (0.7 медленно - 1.3 быстро)
            "pitch": 0.0,           # высота тона (-0.3 низко - +0.3 высоко)
            "energy": 0.0,          # энергия/громкость (-0.3 тихо - +0.3 громко)
            "brightness": 0.0,      # яркость голоса (-0.3 тускло - +0.3 ярко)
            "warmth": 0.0,          # теплота голоса (-0.3 холодно - +0.3 тепло)
            "articulation": 0.0,    # четкость артикуляции (-0.3 нечетко - +0.3 четко)
        }
    
    def apply_preset(self, preset_name: str):
        """Применить предустановку эмоции"""
        presets = {
            "neutral": {
                "tempo": 1.0, "pitch": 0.0, "energy": 0.0,
                "brightness": 0.0, "warmth": 0.0, "articulation": 0.0
            },
            "happy": {
                "tempo": 1.2, "pitch": 0.2, "energy": 0.3,
                "brightness": 0.3, "warmth": 0.2, "articulation": 0.1
            },
            "sad": {
                "tempo": 0.8, "pitch": -0.2, "energy": -0.3,
                "brightness": -0.2, "warmth": -0.1, "articulation": 0.0
            },
            "angry": {
                "tempo": 1.1, "pitch": 0.1, "energy": 0.4,
                "brightness": 0.1, "warmth": -0.2, "articulation": 0.3
            },
            "surprised": {
                "tempo": 1.3, "pitch": 0.4, "energy": 0.5,
                "brightness": 0.4, "warmth": 0.1, "articulation": 0.2
            },
            "energetic": {
                "tempo": 1.4, "pitch": 0.1, "energy": 0.6,
                "brightness": 0.2, "warmth": 0.1, "articulation": 0.1
            },
            "calm": {
                "tempo": 0.9, "pitch": -0.1, "energy": -0.2,
                "brightness": -0.1, "warmth": 0.3, "articulation": 0.0
            },
            "dramatic": {
                "tempo": 1.0, "pitch": 0.3, "energy": 0.4,
                "brightness": 0.2, "warmth": 0.2, "articulation": 0.4
            },
            "professional": {
                "tempo": 1.0, "pitch": 0.0, "energy": 0.1,
                "brightness": 0.0, "warmth": 0.0, "articulation": 0.5
            },
        }
        
        if preset_name in presets:
            self.params.update(presets[preset_name])
            return True
        return False
    
    def to_instruct_string(self) -> str:
        """Преобразовать параметры в текстовую инструкцию"""
        parts = []
        
        # Темп
        if self.params["tempo"] > 1.1:
            parts.append("быстрый темп речи")
        elif self.params["tempo"] < 0.9:
            parts.append("медленный темп речи")
        
        # Высота тона
        if self.params["pitch"] > 0.1:
            parts.append("высокий тон голоса")
        elif self.params["pitch"] < -0.1:
            parts.append("низкий тон голоса")
        
        # Энергия
        if self.params["energy"] > 0.2:
            parts.append("энергичный и громкий голос")
        elif self.params["energy"] < -0.2:
            parts.append("тихий и вялый голос")
        
        # Яркость
        if self.params["brightness"] > 0.2:
            parts.append("яркий и звонкий голос")
        elif self.params["brightness"] < -0.2:
            parts.append("тусклый голос")
        
        # Теплота
        if self.params["warmth"] > 0.2:
            parts.append("тёплый и мягкий голос")
        elif self.params["warmth"] < -0.2:
            parts.append("холодный голос")
        
        # Четкость
        if self.params["articulation"] > 0.3:
            parts.append("очень чёткая дикция")
        elif self.params["articulation"] > 0.1:
            parts.append("чёткая артикуляция")
        
        return ", ".join(parts) if parts else "естественный и нейтральный голос"
    
    def to_dict(self) -> dict:
        """Вернуть параметры как словарь"""
        return self.params.copy()


def parse_text_with_emotions(text: str) -> List[Tuple[str, str]]:
    """
    Разобрать текст с эмоциональными маркерами
    Возвращает список (текст, эмоция)
    """
    # Паттерн для поиска эмоциональных маркеров
    emotion_pattern = r'/(happy|sad|angry|surprised|energetic|calm|dramatic|professional|neutral)\b'
    
    # Находим все маркеры и их позиции
    segments = []
    current_emotion = "neutral"
    current_text = ""
    
    # Разделяем текст по предложениям
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        
        # Проверяем, есть ли в начале предложения маркер эмоции
        match = re.match(r'^/(happy|sad|angry|surprised|energetic|calm|dramatic|professional|neutral)\b\s*', sentence)
        
        if match:
            emotion = match.group(1)
            # Убираем маркер из текста
            sentence_text = sentence[match.end():].strip()
            if sentence_text:
                segments.append((sentence_text, emotion))
                current_emotion = emotion
        else:
            # Используем текущую эмоцию или нейтральную
            segments.append((sentence, current_emotion))
    
    # Если не нашли маркеров, возвращаем весь текст с нейтральной эмоцией
    if not segments:
        segments = [(text, "neutral")]
    
    return segments


def clean_text_from_emotion_markers(text: str) -> str:
    """Очистить текст от маркеров эмоций"""
    # Удаляем все маркеры вида /emotion из текста
    cleaned = re.sub(r'/(happy|sad|angry|surprised|energetic|calm|dramatic|professional|neutral)\b\s*', '', text)
    # Убираем лишние пробелы
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned


class QwenTTSVoiceClone:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "qwen_model": ("QWEN_TTS_MODEL",),
                "text": ("STRING", {
                    "default": "Привет! /happy Это здорово! /sad Но потом стало грустно.",
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
                "global_instruct": ("STRING", {
                    "default": "естественный голос, хорошая дикция",
                    "multiline": True
                }),
                "enable_emotion_parsing": (["enabled", "disabled"], {"default": "enabled"}),
                # Индивидуальные эмоциональные параметры
                "tempo": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.05}),
                "pitch": ("FLOAT", {"default": 0.0, "min": -0.5, "max": 0.5, "step": 0.05}),
                "energy": ("FLOAT", {"default": 0.0, "min": -0.5, "max": 0.5, "step": 0.05}),
                "brightness": ("FLOAT", {"default": 0.0, "min": -0.5, "max": 0.5, "step": 0.05}),
                "warmth": ("FLOAT", {"default": 0.0, "min": -0.5, "max": 0.5, "step": 0.05}),
                "articulation": ("FLOAT", {"default": 0.0, "min": -0.5, "max": 0.5, "step": 0.05}),
            }
        }
    
    RETURN_TYPES = ("AUDIO", "STRING", "DICT")
    RETURN_NAMES = ("audio", "info", "emotion_params")
    FUNCTION = "clone_voice"
    CATEGORY = "audio/tts"
    
    def clone_voice(self, qwen_model, text, ref_audio, ref_text, language, 
                   clone_mode, temperature, seed=0, output_prefix="clone_",
                   global_instruct="естественный голос, хорошая дикция",
                   enable_emotion_parsing="enabled",
                   tempo=1.0, pitch=0.0, energy=0.0, 
                   brightness=0.0, warmth=0.0, articulation=0.0):
        
        model_type = qwen_model.metadata.get("model_type", "base")
        if model_type != "base":
            raise ValueError("Клонирование работает только с моделями типа '-Base'")
        
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        print(f"🎤 Клонирование голоса...")
        
        # Создаем объект управления эмоциональными параметрами
        emotion_control = EmotionControlParameters()
        
        # Устанавливаем индивидуальные параметры
        emotion_control.params.update({
            "tempo": tempo,
            "pitch": pitch,
            "energy": energy,
            "brightness": brightness,
            "warmth": warmth,
            "articulation": articulation,
        })
        
        # Безопасная обработка output_prefix
        safe_output_prefix = PathSanitizer.sanitize_filename(output_prefix)
        
        # Глобальная инструкция
        base_instruct = global_instruct.strip()
        
        # Парсим текст на сегменты с эмоциями
        if enable_emotion_parsing == "enabled":
            segments = parse_text_with_emotions(text)
            print(f"📝 Найдено {len(segments)} сегментов с эмоциями:")
            for i, (seg_text, emotion) in enumerate(segments):
                print(f"  {i+1}. [{emotion}] {seg_text[:60]}...")
        else:
            # Если парсинг отключен, используем весь текст с нейтральной эмоцией
            segments = [(clean_text_from_emotion_markers(text), "neutral")]
            print(f"📝 Используется весь текст с нейтральной эмоцией")
        
        # Если только один сегмент, генерируем целиком
        if len(segments) == 1:
            seg_text, emotion = segments[0]
            
            # Применяем пресет эмоции
            emotion_control.apply_preset(emotion)
            
            # Комбинируем инструкции
            emotion_instruct = emotion_control.to_instruct_string()
            if base_instruct:
                full_instruct = f"{base_instruct}, {emotion_instruct}"
            else:
                full_instruct = emotion_instruct
            
            print(f"🎭 Эмоция: {emotion}")
            print(f"📋 Полная инструкция: {full_instruct}")
            print(f"📝 Текст для произношения: {seg_text[:100]}...")
            
            try:
                result_audio, result_info = self._generate_single_clone(
                    qwen_model, seg_text, ref_audio, ref_text, language,
                    clone_mode, temperature, full_instruct
                )
                
                # Добавляем информацию об эмоциях
                info_dict = json.loads(result_info)
                info_dict["emotion_params"] = emotion_control.to_dict()
                info_dict["emotion_preset"] = emotion
                info_dict["global_instruct"] = global_instruct
                info_dict["spoken_text"] = seg_text
                
                return (
                    result_audio,
                    json.dumps(info_dict, indent=2),
                    emotion_control.to_dict()
                )
                
            except Exception as e:
                print(f"❌ Ошибка генерации: {e}")
                traceback.print_exc()
                silence = torch.zeros((1, 1, 24000))
                sr = 24000
                return (
                    {"waveform": silence, "sample_rate": sr},
                    f'{{"error": "{str(e)}"}}',
                    emotion_control.to_dict()
                )
        
        # Множественные сегменты - генерируем и склеиваем
        else:
            print(f"✂️ Генерация {len(segments)} сегментов с разными эмоциями...")
            
            all_audio_segments = []
            all_segments_info = []
            
            for i, (seg_text, emotion) in enumerate(segments):
                print(f"  Сегмент {i+1}/{len(segments)}: [{emotion}]")
                
                # Применяем пресет для текущей эмоции
                emotion_control.apply_preset(emotion)
                
                # Комбинируем инструкции
                emotion_instruct = emotion_control.to_instruct_string()
                if base_instruct:
                    full_instruct = f"{base_instruct}, {emotion_instruct}"
                else:
                    full_instruct = emotion_instruct
                
                print(f"    📋 Инструкция: {full_instruct}")
                print(f"    📝 Текст: {seg_text[:80]}...")
                
                try:
                    # Генерируем сегмент
                    segment_audio, segment_info = self._generate_single_clone(
                        qwen_model, seg_text, ref_audio, ref_text, language,
                        clone_mode, temperature, full_instruct
                    )
                    
                    all_audio_segments.append(segment_audio)
                    
                    # Сохраняем информацию о сегменте
                    seg_info = {
                        "text": seg_text,
                        "emotion": emotion,
                        "emotion_params": emotion_control.to_dict(),
                        "instruct": full_instruct,
                        "duration": len(segment_audio["waveform"].squeeze()) / segment_audio["sample_rate"]
                    }
                    all_segments_info.append(seg_info)
                    
                    print(f"    ✅ Успешно, длина: {seg_info['duration']:.2f} сек")
                    
                except Exception as e:
                    print(f"    ❌ Ошибка: {e}")
                    # Добавляем тишину вместо ошибки
                    silence = {"waveform": torch.zeros((1, 1, 24000)), "sample_rate": 24000}
                    all_audio_segments.append(silence)
                    
                    seg_info = {
                        "text": seg_text,
                        "emotion": emotion,
                        "emotion_params": emotion_control.to_dict(),
                        "instruct": full_instruct,
                        "error": str(e),
                        "duration": 0
                    }
                    all_segments_info.append(seg_info)
            
            # Склеиваем все сегменты
            print("🔗 Склеивание сегментов...")
            final_audio = self._concatenate_audio_segments(all_audio_segments)
            
            # Формируем итоговую информацию
            total_duration = sum(info.get("duration", 0) for info in all_segments_info)
            
            final_info = {
                "sample_rate": final_audio["sample_rate"],
                "total_duration": total_duration,
                "segments_count": len(segments),
                "segments": all_segments_info,
                "global_instruct": global_instruct,
                "enable_emotion_parsing": enable_emotion_parsing,
                "original_text": text,
                "cleaned_text": " ".join([seg[0] for seg in segments]),
            }
            
            print(f"✅ Все сегменты склеены, общая длина: {total_duration:.2f} сек")
            
            return (
                final_audio,
                json.dumps(final_info, indent=2, ensure_ascii=False),
                emotion_control.to_dict()
            )
    
    def _generate_single_clone(self, qwen_model, text, ref_audio, ref_text, language,
                             clone_mode, temperature, instruct=""):
        """Генерирует один сегмент клонированного голоса"""
        
        # === Извлечение и конвертация референсного аудио ===
        if isinstance(ref_audio, dict) and "waveform" in ref_audio:
            ref_wave = ref_audio["waveform"]  # [B, C, T]
            ref_sr = ref_audio.get("sample_rate", 24000)
        else:
            ref_wave = ref_audio
            ref_sr = 24000

        if isinstance(ref_wave, torch.Tensor):
            ref_np = ref_wave.cpu().numpy()
        else:
            ref_np = np.array(ref_wave)

        # Приведение к форме [C, T]
        if ref_np.ndim == 3:
            ref_np = ref_np[0]  # [B, C, T] → [C, T]
        elif ref_np.ndim == 1:
            ref_np = ref_np[np.newaxis, :]  # [T] → [1, T]

        # Транспонирование в [T, C] для soundfile
        ref_np = ref_np.T

        # Нормализация типа и диапазона
        if ref_np.dtype in [np.int16, np.int32, np.int64]:
            ref_np = ref_np.astype(np.float32) / 32768.0
        elif ref_np.dtype == np.float64:
            ref_np = ref_np.astype(np.float32)

        # Обрезка до [-1, 1]
        ref_np = np.clip(ref_np, -1.0, 1.0)

        # Используем безопасный метод создания временного файла
        temp_ref_path = PathSanitizer.create_secure_temp_file('.wav')
        sf.write(temp_ref_path, ref_np, ref_sr, subtype='FLOAT')
        
        # ВАЖНО: НЕ добавляем инструкцию к тексту!
        # Инструкция передается отдельно в generate_voice_clone
        full_text = text  # Используем только оригинальный текст
        
        print(f"    📤 Передача инструкции отдельно, текст: {full_text[:60]}...")
        
        # Формируем аргументы для клонирования
        gen_kwargs = {
            "text": full_text,  # Только текст для произношения
            "ref_audio": temp_ref_path,
            "ref_text": ref_text,
            "language": language if language != "Auto" else None,
            "x_vector_only_mode": (clone_mode == "x_vector_only"),
            "temperature": temperature,
            "top_p": 0.9,
            "top_k": 50,
            "repetition_penalty": 1.05,
            "max_new_tokens": 1024,
            "do_sample": True,
        }
        
        # Если есть инструкция, добавляем ее в kwargs
        if instruct:
            # Qwen TTS использует параметр 'instruct' для передачи инструкций
            # Но для модели Base может не поддерживаться
            try:
                # Пробуем передать инструкцию через ref_text или в отдельном параметре
                if hasattr(qwen_model, 'generate_voice_clone_with_instruct'):
                    # Если есть специальный метод с инструкцией
                    wavs, sr = qwen_model.generate_voice_clone_with_instruct(
                        **gen_kwargs, instruct=instruct
                    )
                else:
                    # Стандартный способ - инструкция передается в ref_text или text
                    # Для Qwen можно попробовать добавить инструкцию в ref_text
                    original_ref_text = ref_text
                    enhanced_ref_text = f"[{instruct}] {original_ref_text}"
                    gen_kwargs["ref_text"] = enhanced_ref_text
                    
                    start_time = time.time()
                    wavs, sr = qwen_model.generate_voice_clone(**gen_kwargs)
                    generation_time = time.time() - start_time
                    
                    # Возвращаем оригинальный ref_text в info
                    gen_kwargs["ref_text"] = original_ref_text
            except Exception as e:
                print(f"⚠️ Не удалось передать инструкцию, используем базовую генерацию: {e}")
                start_time = time.time()
                wavs, sr = qwen_model.generate_voice_clone(**gen_kwargs)
                generation_time = time.time() - start_time
        else:
            start_time = time.time()
            wavs, sr = qwen_model.generate_voice_clone(**gen_kwargs)
            generation_time = time.time() - start_time
        
        # Безопасное удаление временного файла
        try:
            if os.path.exists(temp_ref_path):
                os.unlink(temp_ref_path)
        except Exception as e:
            print(f"⚠️ Не удалось удалить временный файл {temp_ref_path}: {e}")
        
        audio_data = wavs[0] if isinstance(wavs, list) else wavs
        if torch.is_tensor(audio_data):
            audio_data = audio_data.cpu().numpy()
        
        audio_tensor = torch.from_numpy(audio_data).unsqueeze(0).unsqueeze(0)
        
        info = {
            "sample_rate": sr,
            "duration": len(audio_data) / sr,
            "generation_time": generation_time,
            "clone_mode": clone_mode,
            "temperature": temperature,
            "instruct": instruct if instruct else None,
        }
        
        return {"waveform": audio_tensor, "sample_rate": sr}, json.dumps(info, indent=2)
    
    def _concatenate_audio_segments(self, audio_segments: List[dict]) -> dict:
        """Склеить несколько аудио сегментов в один"""
        if not audio_segments:
            # Возвращаем тишину, если нет сегментов
            return {"waveform": torch.zeros((1, 1, 24000)), "sample_rate": 24000}
        
        # Проверяем, что все сегменты имеют одинаковую частоту дискретизации
        sample_rates = [seg["sample_rate"] for seg in audio_segments]
        if len(set(sample_rates)) > 1:
            print("⚠️ Разные частоты дискретизации, используем первую")
        
        target_sr = sample_rates[0]
        
        # Собираем все waveform в один тензор
        waveforms = []
        for seg in audio_segments:
            wf = seg["waveform"]
            if wf.shape[0] > 1:  # [B, C, T] -> берем первый батч
                wf = wf[0:1]
            waveforms.append(wf)
        
        # Склеиваем по временной оси
        concatenated = torch.cat(waveforms, dim=-1)  # dim=-1 = временная ось
        
        return {"waveform": concatenated, "sample_rate": target_sr}


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
                "language": ("STRING", {"default": "Russian"}),
                "temperature": ("FLOAT", {"default": 0.8, "min": 0.1, "max": 2.0}),
                "separator": ("STRING", {"default": "|"}),
            },
            "optional": {
                "output_prefix": ("STRING", {"default": "batch_"}),
                "seed": ("INT", {"default": 0}),
                "emotion_preset": (["neutral", "happy", "sad", "angry", "surprised", 
                                  "energetic", "calm", "dramatic", "professional"], 
                                 {"default": "neutral"}),
                "instruct": ("STRING", {"default": "", "multiline": True}),
            }
        }
    
    RETURN_TYPES = ("AUDIO", "STRING", "STRING")
    RETURN_NAMES = ("audio", "info", "filenames")
    OUTPUT_IS_LIST = (True, False, False)
    FUNCTION = "batch_generate"
    CATEGORY = "audio/tts"
    
    def batch_generate(self, qwen_model, text_list, language, temperature, separator="|",
                      output_prefix="batch_", seed=0, emotion_preset="neutral", instruct=""):
        
        torch.manual_seed(seed)
        texts = [t.strip() for t in text_list.split(separator) if t.strip()]
        
        print(f"🔄 Пакетная генерация: {len(texts)} текстов")
        print(f"😊 Эмоция: {emotion_preset}")
        
        # Безопасная обработка output_prefix
        safe_output_prefix = PathSanitizer.sanitize_filename(output_prefix)
        
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
        
        audio_outputs = []
        filenames = []
        
        for i, text in enumerate(texts):
            print(f"  {i+1}/{len(texts)}: {text[:40]}...")
            
            try:
                node = QwenTTSGenerate()
                audio_dict, _ = node.generate_speech(
                    qwen_model=qwen_model,
                    text=text,
                    language=language,
                    temperature=temperature,
                    top_p=0.9,
                    seed=seed + i,
                    speaker="Vivian",
                    instruct=instruct,
                    emotion_preset=emotion_preset
                )
                audio_outputs.append(audio_dict)
                filename = f"{safe_output_prefix}{i+1:03d}.wav"
                filenames.append(filename)
                
            except Exception as e:
                print(f"  ❌ Ошибка в тексте {i+1}: {e}")
                silence = {"waveform": torch.zeros((1, 1, 24000)), "sample_rate": 24000}
                audio_outputs.append(silence)
                filenames.append(f"error_{i+1}.wav")
        
        info = f"Сгенерировано {len(audio_outputs)} аудиофайлов, эмоция: {emotion_preset}"
        filenames_str = separator.join(filenames)
        
        return (audio_outputs, info, filenames_str)


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
                "clear_output_dir": (["no", "yes"], {"default": "no"}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("filepath",)
    FUNCTION = "save_audio"
    CATEGORY = "audio/tts"
    
    def save_audio(self, audio, filename, sample_rate, output_dir="output/tts", 
                   metadata="", clear_output_dir="no"):
        
        # === БЕЗОПАСНАЯ ОБРАБОТКА output_dir ===
        # 1. Санитаризируем output_dir
        safe_output_dir = PathSanitizer.sanitize_path(output_dir)
        if not safe_output_dir:
            safe_output_dir = "output/tts"
        
        # 2. Получаем базовую директорию вывода из ComfyUI
        base_output_dir = folder_paths.get_output_directory()
        
        # 3. Создаем полный путь и нормализуем его
        full_output_dir = os.path.join(base_output_dir, safe_output_dir)
        full_output_dir = os.path.normpath(full_output_dir)
        
        # 4. Проверяем, что путь остается внутри разрешенной директории
        base_output_norm = os.path.normpath(base_output_dir)
        if not full_output_dir.startswith(base_output_norm):
            print(f"⚠️ Опасный путь output_dir: {output_dir}, используется стандартный")
            safe_output_dir = "output/tts"
            full_output_dir = os.path.join(base_output_dir, safe_output_dir)
            full_output_dir = os.path.normpath(full_output_dir)
        
        # === БЕЗОПАСНАЯ ОБРАБОТКА filename ===
        # 1. Санитаризируем имя файла
        safe_filename = PathSanitizer.sanitize_filename(filename)
        
        # 2. Убеждаемся, что у файла безопасное расширение
        safe_filename = PathSanitizer.ensure_safe_extension(safe_filename, ".wav")
        
        # 3. Формируем полный путь к файлу
        filepath = os.path.join(full_output_dir, safe_filename)
        filepath = os.path.normpath(filepath)
        
        # 4. Двойная проверка, что файл сохраняется в правильной директории
        if not filepath.startswith(full_output_dir):
            print(f"⚠️ Попытка сохранить файл за пределами директории: {filename}")
            safe_filename = "output.wav"
            filepath = os.path.join(full_output_dir, safe_filename)
        
        # === ОЧИСТКА ДИРЕКТОРИИ (если выбрано) ===
        if clear_output_dir == "yes":
            # Очищаем только если директория существует и внутри разрешенной зоны
            if os.path.exists(full_output_dir) and full_output_dir.startswith(base_output_norm):
                try:
                    print(f"🧹 Очистка директории: {full_output_dir}")
                    for item in os.listdir(full_output_dir):
                        item_path = os.path.join(full_output_dir, item)
                        try:
                            if os.path.isfile(item_path) or os.path.islink(item_path):
                                os.unlink(item_path)
                            elif os.path.isdir(item_path):
                                # Проверяем, что это не симлинк на внешнюю директорию
                                if not os.path.islink(item_path):
                                    shutil.rmtree(item_path)
                        except Exception as e:
                            print(f"⚠️ Не удалось удалить {item_path}: {e}")
                except Exception as e:
                    print(f"⚠️ Ошибка при очистке директории: {e}")
            else:
                print(f"⚠️ Директория для очистки не найдена или небезопасна: {full_output_dir}")
        
        # === СОЗДАНИЕ ДИРЕКТОРИИ ===
        try:
            os.makedirs(full_output_dir, exist_ok=True)
            print(f"📁 Директория создана/подготовлена: {full_output_dir}")
        except Exception as e:
            print(f"❌ Не удалось создать директорию {full_output_dir}: {e}")
            # Создаем стандартную директорию в случае ошибки
            full_output_dir = os.path.join(base_output_dir, "output")
            os.makedirs(full_output_dir, exist_ok=True)
            safe_filename = "output.wav"
            filepath = os.path.join(full_output_dir, safe_filename)
        
        # === СОХРАНЕНИЕ АУДИО ===
        try:
            if isinstance(audio, dict) and "waveform" in audio:
                audio_np = audio["waveform"].cpu().numpy().squeeze()
                actual_sample_rate = audio.get("sample_rate", sample_rate)
            else:
                audio_np = audio.cpu().numpy().squeeze() if isinstance(audio, torch.Tensor) else np.array(audio)
                actual_sample_rate = sample_rate
            
            # Используем актуальную частоту дискретизации из аудио или переданную
            save_sample_rate = actual_sample_rate if actual_sample_rate else sample_rate
            
            # Нормализуем аудио данные
            if audio_np.dtype != np.float32:
                if audio_np.dtype in [np.int16, np.int32]:
                    audio_np = audio_np.astype(np.float32) / 32768.0
                else:
                    audio_np = audio_np.astype(np.float32)
            
            # Обрезаем значения до безопасного диапазона
            audio_np = np.clip(audio_np, -1.0, 1.0)
            
            # Сохраняем файл
            sf.write(filepath, audio_np, save_sample_rate, subtype='FLOAT')
            print(f"💾 Аудио сохранено: {filepath}")
            print(f"   Частота дискретизации: {save_sample_rate} Hz")
            print(f"   Длина: {len(audio_np)/save_sample_rate:.2f} сек")
            
        except Exception as e:
            print(f"❌ Ошибка при сохранении аудио: {e}")
            traceback.print_exc()
            # Возвращаем путь к файлу даже при ошибке
            return (filepath,)
        
        # === СОХРАНЕНИЕ МЕТАДАННЫХ ===
        if metadata and isinstance(metadata, str) and metadata.strip():
            metadata_file = os.path.splitext(filepath)[0] + '.json'
            try:
                # Пытаемся разобрать JSON
                if metadata.strip().startswith('{'):
                    meta_dict = json.loads(metadata)
                else:
                    meta_dict = {"metadata": metadata}
                
                # Добавляем системную информацию
                meta_dict.update({
                    "save_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "output_dir": safe_output_dir,
                    "filename": safe_filename,
                    "cleared_before_save": clear_output_dir == "yes",
                    "sample_rate": save_sample_rate,
                    "duration_sec": len(audio_np)/save_sample_rate if 'audio_np' in locals() else 0,
                })
                
                # Сохраняем метаданные
                with open(metadata_file, 'w', encoding='utf-8') as f:
                    json.dump(meta_dict, f, indent=2, ensure_ascii=False)
                
                print(f"📋 Метаданные сохранены: {metadata_file}")
                
            except json.JSONDecodeError:
                # Если это не JSON, сохраняем как текст
                try:
                    with open(metadata_file, 'w', encoding='utf-8') as f:
                        f.write(f"Metadata saved at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                        f.write("=" * 50 + "\n")
                        f.write(metadata)
                    print(f"📝 Метаданные сохранены как текст: {metadata_file}")
                except Exception as e:
                    print(f"⚠️ Не удалось сохранить метаданные: {e}")
            except Exception as e:
                print(f"⚠️ Ошибка при сохранении метаданных: {e}")
        
        return (filepath,)


class QwenTTSEmotionMixer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_tensors": ("AUDIO",),
                "weights": ("STRING", {"default": "1.0,0.5,0.3", "multiline": False}),
                "normalize": (["yes", "no"], {"default": "yes"}),
            }
        }
    
    INPUT_IS_LIST = True
    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "mix_emotions"
    CATEGORY = "audio/tts"
    
    def mix_emotions(self, audio_tensors, weights, normalize):
        print("🎚️ Микширование эмоциональных вариантов...")
        
        try:
            # Безопасная обработка весов
            if weights and isinstance(weights, list) and len(weights) > 0:
                weight_str = str(weights[0])
                # Разрешаем только цифры, точки и запятые
                safe_weight_str = re.sub(r'[^0-9.,]', '', weight_str)
                weight_list = [float(w.strip()) for w in safe_weight_str.split(',') if w.strip()]
            else:
                weight_list = []
        except Exception as e:
            print(f"⚠️ Ошибка при обработке весов: {e}")
            weight_list = [1.0] * len(audio_tensors)
        
        # Если весов меньше чем аудио, дополняем
        if len(weight_list) < len(audio_tensors):
            weight_list.extend([1.0] * (len(audio_tensors) - len(weight_list)))
        
        waveforms = []
        max_len = 0
        for a in audio_tensors:
            if isinstance(a, dict) and "waveform" in a:
                wf = a["waveform"].cpu().numpy().squeeze()
            else:
                wf = a.cpu().numpy().squeeze() if isinstance(a, torch.Tensor) else np.array(a)
            waveforms.append(wf)
            max_len = max(max_len, len(wf))
        
        padded = []
        for wf in waveforms:
            if len(wf) < max_len:
                wf = np.pad(wf, (0, max_len - len(wf)))
            else:
                wf = wf[:max_len]
            padded.append(wf)
        
        if normalize == "yes" and weight_list:
            weight_sum = sum(weight_list)
            if weight_sum > 0:
                weight_list = [w / weight_sum for w in weight_list]
        
        mixed = np.zeros_like(padded[0])
        for wf, w in zip(padded, weight_list):
            mixed += wf * w
        
        # Нормализуем результат
        max_val = np.max(np.abs(mixed))
        if max_val > 1.0:
            mixed = mixed / max_val
        
        mixed_tensor = torch.from_numpy(mixed).unsqueeze(0).unsqueeze(0)
        sample_rate = audio_tensors[0].get("sample_rate", 24000) if isinstance(audio_tensors[0], dict) else 24000
        
        print(f"✅ Смешано {len(audio_tensors)} аудио с весами {weight_list}")
        return ({"waveform": mixed_tensor, "sample_rate": sample_rate},)


class EmotionParametersPreview:
    """Нода для предпросмотра эмоциональных параметров"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "emotion_preset": (["neutral", "happy", "sad", "angry", "surprised", 
                                  "energetic", "calm", "dramatic", "professional"], 
                                 {"default": "neutral"}),
            },
            "optional": {
                "tempo": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.05}),
                "pitch": ("FLOAT", {"default": 0.0, "min": -0.5, "max": 0.5, "step": 0.05}),
                "energy": ("FLOAT", {"default": 0.0, "min": -0.5, "max": 0.5, "step": 0.05}),
                "brightness": ("FLOAT", {"default": 0.0, "min": -0.5, "max": 0.5, "step": 0.05}),
                "warmth": ("FLOAT", {"default": 0.0, "min": -0.5, "max": 0.5, "step": 0.05}),
                "articulation": ("FLOAT", {"default": 0.0, "min": -0.5, "max": 0.5, "step": 0.05}),
            }
        }
    
    RETURN_TYPES = ("DICT", "STRING")
    RETURN_NAMES = ("emotion_params", "instruct")
    FUNCTION = "preview_params"
    CATEGORY = "audio/tts/emotion"
    
    def preview_params(self, emotion_preset="neutral", tempo=1.0, pitch=0.0, 
                      energy=0.0, brightness=0.0, warmth=0.0, articulation=0.0):
        
        emotion_control = EmotionControlParameters()
        
        # Применяем пресет
        emotion_control.apply_preset(emotion_preset)
        
        # Перезаписываем индивидуальными параметрами
        emotion_control.params.update({
            "tempo": tempo,
            "pitch": pitch,
            "energy": energy,
            "brightness": brightness,
            "warmth": warmth,
            "articulation": articulation,
        })
        
        # Генерируем инструкцию
        instruct = emotion_control.to_instruct_string()
        
        print(f"🎭 Пресет: {emotion_preset}")
        print(f"📊 Параметры: {emotion_control.params}")
        print(f"📋 Инструкция: {instruct}")
        
        return (emotion_control.to_dict(), instruct)


# === РЕГИСТРАЦИЯ НОД ===
NODE_CLASS_MAPPINGS = {
    "QwenTTSModelLoader": QwenTTSModelLoader,
    "QwenTTSGenerate": QwenTTSGenerate,
    "QwenTTSVoiceClone": QwenTTSVoiceClone,
    "QwenTTSBatchGenerate": QwenTTSBatchGenerate,
    "QwenTTSAudioSaver": QwenTTSAudioSaver,
    "QwenTTSEmotionMixer": QwenTTSEmotionMixer,
    "EmotionParametersPreview": EmotionParametersPreview,
}

# Добавляем префикс "DVA" ко всем отображаемым именам
NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenTTSModelLoader": "DVA 🤖 Qwen TTS Loader",
    "QwenTTSGenerate": "DVA 🎤 Qwen TTS Generate",
    "QwenTTSVoiceClone": "DVA 🎭 Qwen TTS Voice Clone",
    "QwenTTSBatchGenerate": "DVA 📚 Qwen TTS Batch Generate",
    "QwenTTSAudioSaver": "DVA 💾 Qwen TTS Audio Saver",
    "QwenTTSEmotionMixer": "DVA 🔀 Qwen TTS Emotion Mixer",
    "EmotionParametersPreview": "DVA 🎭 Emotion Parameters",
}
