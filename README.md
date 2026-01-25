Qwen-TTS Nodes for ComfyUI



Ноды для интеграции Qwen3-TTS в ComfyUI с поддержкой эмоций и клонирования голоса.



\## 🚀 Установка



1\. Скопируйте папку `qwen_tts_comfy` в `ComfyUI/custom_nodes/` или через git clone

2\. Установите зависимости:   pip install qwen-tts soundfile или через requirements.txt


📦 DVA 🤖 Qwen TTS Loader
Загружает модель синтеза речи Qwen-TTS в память.
Поддерживает модели:

Base — для клонирования голоса,
CustomVoice — управляемый синтез по имени спикера (например, Vivian),
VoiceDesign — синтез по текстовому описанию голоса.
Позволяет выбрать точность (fp16, bf16, fp32), устройство (cuda/cpu) и тип внимания.

🎤 DVA 🎤 Qwen TTS Generate
Генерирует речь из текста без референсного аудио.
Работает только с моделями CustomVoice или VoiceDesign.
Поддерживает:выбор языка (русский, английский и др.),
управление эмоцией через пресеты (нейтрально, радостно, грустно и т.д.),
ручную настройку температуры, top-p и других параметров.
Для CustomVoice можно указать имя спикера (например, Vivian).

🎭 DVA 🎭 Qwen TTS Voice Clone
Клонирует голос из референсного аудиофайла.
Работает только с моделями типа -Base.
Требует:входное аудио (референс),
текст, произнесённый в этом аудио (ref_text),
новый текст для синтеза.
Идеально для воссоздания уникального голоса.

📚 DVA 📚 Qwen TTS Batch Generate
Генерирует несколько аудиофайлов за один запуск.
Разделяет входной текст по указанному разделителю (например, |) и синтезирует каждую часть отдельно.
Полезно для создания диалогов, списков фраз или тестовых наборов.

💾 DVA 💾 Qwen TTS Audio Saver
Сохраняет сгенерированное аудио на диск в формате .wav.
Позволяет указать: имя файла,
выходную папку (по умолчанию — output/tts),
дополнительные метаданные (сохраняются в .json).
Автоматически очищает недопустимые символы из имени файла.

🔀 DVA 🔀 Qwen TTS Emotion Mixer
Смешивает несколько вариантов одного и того же текста, синтезированных с разными эмоциями или параметрами.
Например: 70% «спокойного» + 30% «энергичного» голоса.
Можно задать веса вручную и включить нормализацию суммы весов до 1.0.





Qwen-TTS Nodes for ComfyUI

Nodes for integrating Qwen3-TTS into ComfyUI with support for emotions and voice cloning.

## 🚀 Installation

1.Copy the qwen_tts_comfy folder to ComfyUI/custom_nodes/ or via git clone

2.Install dependencies: pip install qwen-tts soundfile or via requirements.txt

📦 DVA 🤖 Qwen TTS Loader Loads the Qwen-TTS speech synthesis model into memory.Supports models:

Base - for voice cloning,CustomVoice - controlled synthesis by speaker name (for example,Vivian),VoiceDesign - synthesis based on a text description of a voice.Allows you to select accuracy (fp16,bf16,fp32),device (cuda/cpu) and attention type.

🎤 DVA 🎤 Qwen TTS Generate Generates speech from text without reference audio.Works only with CustomVoice or VoiceDesign models.Supports: language selection (Russian,English, etc.),emotion control through presets (neutral,joyfullysad, etc.d.),manual temperature setting,top-p and other parameters.For CustomVoice, you can specify the name of the speaker (for example,Vivian).

🎭 DVA 🎭 Qwen TTS Voice Clone Clones a voice from a reference audio file.Works only with -Base type models.Requires: audio input (reference),text,spoken in this audio (ref_text),new text for synthesis.Ideal for recreating a unique voice.

📚 DVA 📚 Qwen TTS Batch Generate Generates multiple audio files in one run.Splits input text at the specified delimiter (for example,|) and synthesizes each part separately.Useful for creating






