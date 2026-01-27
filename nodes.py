import os
import shutil  # Добавлен импорт shutil для очистки директории
import torch
import numpy as np
import folder_paths
import json
import time
import traceback
import tempfile
import soundfile as sf

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
            
            if cache_dir and os.path.exists(cache_dir):
                kwargs["cache_dir"] = cache_dir
            
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
        
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        print(f"🎤 Клонирование голоса...")
        print(f"📝 Текст: {text[:50]}...")
        print(f"🔊 Референс: {ref_text[:50]}...")
        print(f"🎭 Режим: {clone_mode}")
        
        try:
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

            # Сохраняем во временный файл
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                temp_ref_path = tmp.name
                sf.write(temp_ref_path, ref_np, ref_sr, subtype='FLOAT')
            
            gen_kwargs = {
                "text": text,
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
            
            start_time = time.time()
            wavs, sr = qwen_model.generate_voice_clone(**gen_kwargs)
            generation_time = time.time() - start_time
            
            try:
                os.unlink(temp_ref_path)
            except:
                pass
            
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
                "original_duration": len(ref_np) / ref_sr,
            }
            
            print(f"✅ Клонирование успешно за {generation_time:.2f} сек")
            print(f"📊 Длительность: {info['duration']:.2f} сек")
            
            return ({"waveform": audio_tensor, "sample_rate": sr}, json.dumps(info, indent=2))
            
        except Exception as e:
            print(f"❌ Ошибка клонирования: {e}")
            traceback.print_exc()
            silence = torch.zeros((1, 1, 24000))
            sr = 24000
            return ({"waveform": silence, "sample_rate": sr}, f"Error: {str(e)}")


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
            }
        }
    
    RETURN_TYPES = ("AUDIO", "STRING", "STRING")
    RETURN_NAMES = ("audio", "info", "filenames")
    OUTPUT_IS_LIST = (True, False, False)
    FUNCTION = "batch_generate"
    CATEGORY = "audio/tts"
    
    def batch_generate(self, qwen_model, text_list, language, temperature, separator="|",
                      output_prefix="batch_", seed=0):
        
        torch.manual_seed(seed)
        texts = [t.strip() for t in text_list.split(separator) if t.strip()]
        
        print(f"🔄 Пакетная генерация: {len(texts)} текстов")
        
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
                    instruct="",
                    emotion_preset="neutral"
                )
                audio_outputs.append(audio_dict)
                filename = f"{output_prefix}{i+1:03d}.wav"
                filenames.append(filename)
                
            except Exception as e:
                print(f"  ❌ Ошибка в тексте {i+1}: {e}")
                silence = {"waveform": torch.zeros((1, 1, 24000)), "sample_rate": 24000}
                audio_outputs.append(silence)
                filenames.append(f"error_{i+1}.wav")
        
        info = f"Сгенерировано {len(audio_outputs)} аудиофайлов"
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
        
        # Получаем полный путь к директории
        full_output_dir = os.path.join(folder_paths.get_output_directory(), output_dir)
        
        # Очищаем директорию, если выбрано "yes"
        if clear_output_dir == "yes" and os.path.exists(full_output_dir):
            try:
                print(f"🧹 Очистка директории: {full_output_dir}")
                for filename in os.listdir(full_output_dir):
                    file_path = os.path.join(full_output_dir, filename)
                    try:
                        if os.path.isfile(file_path) or os.path.islink(file_path):
                            os.unlink(file_path)
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)
                    except Exception as e:
                        print(f"⚠️ Не удалось удалить {file_path}: {e}")
            except Exception as e:
                print(f"⚠️ Ошибка при очистке директории: {e}")
        
        # Создаем директорию
        os.makedirs(full_output_dir, exist_ok=True)
        
        # Обрабатываем имя файла
        safe_filename = "".join(c for c in filename if c.isalnum() or c in (' ', '.', '_', '-')).rstrip()
        if not safe_filename.endswith('.wav'):
            safe_filename += '.wav'
        
        filepath = os.path.join(full_output_dir, safe_filename)
        
        # Сохраняем аудио
        if isinstance(audio, dict) and "waveform" in audio:
            audio_np = audio["waveform"].cpu().numpy().squeeze()
        else:
            audio_np = audio.cpu().numpy().squeeze() if isinstance(audio, torch.Tensor) else np.array(audio)
        
        sf.write(filepath, audio_np, sample_rate)
        
        # Сохраняем метаданные
        if metadata.strip():
            metadata_file = filepath.replace('.wav', '.json')
            try:
                meta_dict = json.loads(metadata) if isinstance(metadata, str) else metadata
                meta_dict.update({
                    "save_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "output_dir": output_dir,
                    "cleared_before_save": clear_output_dir == "yes"
                })
                with open(metadata_file, 'w', encoding='utf-8') as f:
                    json.dump(meta_dict, f, indent=2, ensure_ascii=False)
            except Exception as e:
                print(f"⚠️ Не удалось сохранить метаданные: {e}")
        
        print(f"💾 Аудио сохранено: {filepath}")
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
            weight_list = [float(w.strip()) for w in weights[0].split(',')]
        except:
            weight_list = [1.0] * len(audio_tensors)
        
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
            weight_list = [w / weight_sum for w in weight_list]
        
        mixed = np.zeros_like(padded[0])
        for wf, w in zip(padded, weight_list):
            mixed += wf * w
        
        mixed_tensor = torch.from_numpy(mixed).unsqueeze(0).unsqueeze(0)
        sample_rate = audio_tensors[0].get("sample_rate", 24000) if isinstance(audio_tensors[0], dict) else 24000
        
        print(f"✅ Смешано {len(audio_tensors)} аудио с весами {weight_list}")
        return ({"waveform": mixed_tensor, "sample_rate": sample_rate},)


# === РЕГИСТРАЦИЯ НОД ===
NODE_CLASS_MAPPINGS = {
    "QwenTTSModelLoader": QwenTTSModelLoader,
    "QwenTTSGenerate": QwenTTSGenerate,
    "QwenTTSVoiceClone": QwenTTSVoiceClone,
    "QwenTTSBatchGenerate": QwenTTSBatchGenerate,
    "QwenTTSAudioSaver": QwenTTSAudioSaver,
    "QwenTTSEmotionMixer": QwenTTSEmotionMixer,
}

# Добавляем префикс "DVA" ко всем отображаемым именам
NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenTTSModelLoader": "DVA 🤖 Qwen TTS Loader",
    "QwenTTSGenerate": "DVA 🎤 Qwen TTS Generate",
    "QwenTTSVoiceClone": "DVA 🎭 Qwen TTS Voice Clone",
    "QwenTTSBatchGenerate": "DVA 📚 Qwen TTS Batch Generate",
    "QwenTTSAudioSaver": "DVA 💾 Qwen TTS Audio Saver",
    "QwenTTSEmotionMixer": "DVA 🔀 Qwen TTS Emotion Mixer",
}
