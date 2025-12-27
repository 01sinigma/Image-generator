"""
Веб-интерфейс для Image Generator на базе Gradio
Поддерживает Z-Image-Turbo для генерации и Qwen-Image-Edit-2511 для редактирования изображений

Новые функции:
- Удаление и замена фона (RMBG-2.0)
- LoRA для ускорения генерации (Lightning 4-step)
- Uncensored режим
- Pipeline обработки изображений

Для оптимизации памяти CUDA рекомендуется запускать с переменной окружения:
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python app.py
Или использовать start_app.bat
"""

import gradio as gr
import yaml
from pathlib import Path
from generator import ZImageGenerator
from models import ModelFactory
import logging
import traceback
from datetime import datetime
from PIL import Image
from typing import Optional, Tuple

# Настройка логирования с детальной информацией
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Импорт проверки статуса
try:
    from system_status import SystemStatusChecker
    STATUS_CHECKER_AVAILABLE = True
except ImportError:
    STATUS_CHECKER_AVAILABLE = False
    SystemStatusChecker = None
    logger.warning("system_status модуль недоступен, статус-проверка отключена")

# Импорт удаления фона
try:
    from background_remover import BackgroundRemover, BackgroundPresets
    BACKGROUND_REMOVER_AVAILABLE = True
except ImportError:
    BACKGROUND_REMOVER_AVAILABLE = False
    BackgroundRemover = None
    BackgroundPresets = None
    logger.warning("background_remover модуль недоступен")

# Импорт Pipeline
try:
    from image_pipeline import ImagePipeline, PipelineMode
    PIPELINE_AVAILABLE = True
except ImportError:
    PIPELINE_AVAILABLE = False
    ImagePipeline = None
    PipelineMode = None
    logger.warning("image_pipeline модуль недоступен")

# Импорт LoRA Manager
try:
    from lora_manager import LoRAManager
    LORA_MANAGER_AVAILABLE = True
except ImportError:
    LORA_MANAGER_AVAILABLE = False
    LoRAManager = None
    logger.warning("lora_manager модуль недоступен")

# Импорт ComfyUI API
try:
    from comfyui_api import ComfyUIClient, ComfyUIGGUFGenerator, check_comfyui_status
    COMFYUI_AVAILABLE = True
except ImportError:
    COMFYUI_AVAILABLE = False
    ComfyUIClient = None
    ComfyUIGGUFGenerator = None
    logger.warning("comfyui_api модуль недоступен")

# Глобальные переменные для новых компонентов
background_remover: Optional[BackgroundRemover] = None
image_pipeline: Optional[ImagePipeline] = None
comfyui_generator: Optional[object] = None  # ComfyUIGGUFGenerator

# Информация о моделях и их возможностях
MODEL_INFO = {
    "z-image-turbo": {
        "name": "⚡ Z-Image-Turbo",
        "type": "diffusers",
        "capabilities": ["generate"],
        "description": "Быстрая генерация (5-10 сек, 9 шагов)",
        "recommended_settings": {
            "height": 1024, "width": 1024,
            "steps": 9, "guidance": 0.0
        },
        "features": {
            "fast_generation": True,
            "image_editing": False,
            "lora_support": False,
            "uncensored": False,
        }
    },
    "sdxl-turbo": {
        "name": "🎨 SDXL Turbo",
        "type": "diffusers",
        "capabilities": ["generate"],
        "description": "Универсальная, LoRA, без цензуры (4 шага)",
        "recommended_settings": {
            "height": 512, "width": 512,
            "steps": 4, "guidance": 0.0
        },
        "features": {
            "fast_generation": True,
            "image_editing": False,
            "lora_support": True,
            "uncensored": True,
        }
    },
    "pony-diffusion": {
        "name": "🦄 Pony Diffusion V6",
        "type": "diffusers",
        "capabilities": ["generate"],
        "description": "Аниме/NSFW, LoRA (25 шагов)",
        "recommended_settings": {
            "height": 1024, "width": 1024,
            "steps": 25, "guidance": 7.0
        },
        "features": {
            "fast_generation": False,
            "image_editing": False,
            "lora_support": True,
            "uncensored": True,
            "anime": True,
        }
    },
    "realvis-xl": {
        "name": "📷 RealVisXL V4",
        "type": "diffusers",
        "capabilities": ["generate"],
        "description": "Фотореализм, LoRA (25 шагов)",
        "recommended_settings": {
            "height": 1024, "width": 1024,
            "steps": 25, "guidance": 5.0
        },
        "features": {
            "fast_generation": False,
            "image_editing": False,
            "lora_support": True,
            "uncensored": True,
            "realistic": True,
        }
    },
    "instruct-pix2pix": {
        "name": "🖌️ InstructPix2Pix",
        "type": "diffusers",
        "capabilities": ["edit"],
        "description": "Редактирование по тексту: 'сделай волосы синими' (20 шагов)",
        "recommended_settings": {
            "height": 512, "width": 512,
            "steps": 20, "guidance": 7.5, "image_guidance": 1.5
        },
        "features": {
            "fast_generation": True,
            "image_editing": True,
            "lora_support": False,
            "uncensored": True,
        }
    },
    "sdxl-inpainting": {
        "name": "🎭 SDXL Inpainting",
        "type": "diffusers",
        "capabilities": ["inpaint"],
        "description": "Закрашивание/замена части изображения (25 шагов)",
        "recommended_settings": {
            "height": 1024, "width": 1024,
            "steps": 25, "guidance": 7.5
        },
        "features": {
            "fast_generation": False,
            "image_editing": True,
            "inpainting": True,
            "lora_support": True,
            "uncensored": True,
        }
    },
    "qwen-image-edit": {
        "name": "✨ Qwen-Image-Edit",
        "type": "diffusers",
        "capabilities": ["generate", "edit"],
        "description": "Мощное редактирование (1-3 мин, 40 шагов, нужно 32GB RAM)",
        "recommended_settings": {
            "height": 512, "width": 512,
            "steps": 40, "guidance": 1.0, "true_cfg": 4.0
        },
        "features": {
            "fast_generation": False,
            "image_editing": True,
            "lora_support": True,
            "uncensored": False,
        }
    },
}


def load_config():
    """Загрузка конфигурации для веб-интерфейса"""
    try:
        with open("config.yaml", 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logger.warning(f"Не удалось загрузить конфигурацию: {e}")
        return {}


# Глобальные переменные
generators = {}  # Словарь загруженных генераторов
config = load_config()


def get_available_models():
    """Получение списка доступных моделей"""
    models = []
    
    # Модели из ModelFactory (diffusers)
    try:
        models.extend(ModelFactory.get_available_models())
    except:
        models.append('z-image-turbo')
    
    # ComfyUI GGUF модель (если ComfyUI доступен)
    if COMFYUI_AVAILABLE:
        models.append('qwen-gguf-comfyui')
    
    return models


def get_model_info(model_type: str) -> dict:
    """Получение информации о модели"""
    return MODEL_INFO.get(model_type, {
        "name": model_type,
        "type": "unknown",
        "capabilities": ["generate"],
        "description": "Неизвестная модель",
        "recommended_settings": {"height": 512, "width": 512, "steps": 20, "guidance": 7.0},
        "features": {}
    })


def check_comfyui_available() -> Tuple[bool, str]:
    """Проверка доступности ComfyUI"""
    if not COMFYUI_AVAILABLE:
        return False, "ComfyUI API модуль не установлен"
    
    try:
        status = check_comfyui_status()
        if status.get('available'):
            return True, "ComfyUI доступен и готов к работе"
        else:
            return False, "ComfyUI не запущен. Запустите: start_comfyui.bat"
    except Exception as e:
        return False, f"Ошибка проверки ComfyUI: {str(e)}"


def load_comfyui_model():
    """Загрузка ComfyUI GGUF генератора"""
    global comfyui_generator, generators
    
    if not COMFYUI_AVAILABLE:
        return "❌ ComfyUI API недоступен"
    
    available, message = check_comfyui_available()
    if not available:
        return f"❌ {message}"
    
    try:
        comfyui_generator = ComfyUIGGUFGenerator(
            host="127.0.0.1",
            port=8188,
            model_name="qwen-image-edit-2511-Q4_K_M.gguf"
        )
        generators['qwen-gguf-comfyui'] = comfyui_generator
        logger.info("✅ ComfyUI GGUF генератор подключен")
        return "✅ ComfyUI GGUF готов к работе!"
    except Exception as e:
        logger.error(f"Ошибка подключения ComfyUI: {e}")
        return f"❌ Ошибка: {str(e)[:100]}"


def generate_with_comfyui(
    prompt: str,
    height: int = 512,
    width: int = 512,
    steps: int = 4,
    guidance: float = 1.0,
    seed: int = -1,
    progress=gr.Progress()
):
    """Генерация через ComfyUI GGUF"""
    global comfyui_generator
    
    if comfyui_generator is None:
        load_result = load_comfyui_model()
        if "❌" in load_result:
            gr.Error(load_result)
            return None, load_result
    
    try:
        progress(0.1, desc="Отправка в ComfyUI...")
        
        image = comfyui_generator.generate(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=steps,
            guidance_scale=guidance,
            seed=seed if seed >= 0 else None
        )
        
        if image:
            progress(1.0, desc="Готово!")
            gr.Success("Изображение сгенерировано через ComfyUI!")
            return image, "✅ Сгенерировано через ComfyUI GGUF!"
        else:
            gr.Error("ComfyUI не вернул изображение")
            return None, "❌ Ошибка генерации"
            
    except Exception as e:
        logger.error(f"Ошибка ComfyUI генерации: {e}")
        gr.Error(f"Ошибка: {str(e)[:100]}")
        return None, f"❌ {str(e)[:100]}"


def load_model(model_type: str, show_notification: bool = True):
    """Загрузка модели указанного типа"""
    global generators
    
    logger.info("="*60)
    logger.info(f"🔄 ЗАГРУЗКА МОДЕЛИ: {model_type}")
    logger.info("="*60)
    
    # Специальная обработка для ComfyUI
    if model_type == 'qwen-gguf-comfyui':
        return load_comfyui_model()
    
    if model_type in generators:
        logger.info(f"✅ Модель {model_type} уже загружена")
        logger.info(f"📦 Статус: Готова к работе")
        
        # Проверка функций
        if STATUS_CHECKER_AVAILABLE:
            checker = SystemStatusChecker()
            functions = checker.check_functions(generators)
            if model_type in functions:
                funcs = functions[model_type]
                logger.info(f"🔧 Доступные функции:")
                logger.info(f"   - generate: {'✅' if funcs.get('generate') else '❌'}")
                logger.info(f"   - edit: {'✅' if funcs.get('edit') else '❌'}")
        
        message = f"✅ Модель {model_type} готова к работе!"
        if show_notification:
            gr.Info(f"Модель {model_type} уже загружена и готова к работе")
        return message
    
    try:
        logger.info(f"📥 Начало загрузки модели {model_type}...")
        logger.info(f"⏰ Время начала: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Предупреждение для больших моделей
        if model_type == 'qwen-image-edit':
            logger.warning("⚠️ ВНИМАНИЕ: Qwen-Image-Edit-2511 - очень большая модель (~50GB)")
            logger.warning("⏱️ Загрузка может занять 10-30 минут в зависимости от скорости интернета")
            logger.warning("💡 Рекомендуется скачать модель заранее через: python download_models.py --model qwen-image-edit")
            if show_notification:
                gr.Warning("Qwen-Image-Edit-2511 - большая модель (~50GB). Загрузка может занять 10-30 минут.")
        
        generator = ZImageGenerator(model_type=model_type)
        generators[model_type] = generator
        
        logger.info("="*60)
        logger.info(f"✅ МОДЕЛЬ {model_type} УСПЕШНО ЗАГРУЖЕНА")
        logger.info("="*60)
        logger.info(f"📦 Статус: Готова к работе")
        logger.info(f"🎮 Устройство: {generator.device}")
        
        # Проверка функций после загрузки
        if STATUS_CHECKER_AVAILABLE:
            checker = SystemStatusChecker()
            functions = checker.check_functions(generators)
            if model_type in functions:
                funcs = functions[model_type]
                logger.info(f"🔧 Доступные функции:")
                logger.info(f"   - generate: {'✅ Доступна' if funcs.get('generate') else '❌ Недоступна'}")
                logger.info(f"   - edit: {'✅ Доступна' if funcs.get('edit') else '❌ Недоступна'}")
        
        message = f"✅ Модель {model_type} готова к работе!"
        if show_notification:
            gr.Info(f"Модель {model_type} успешно загружена и готова к работе!")
        return message
        
    except KeyboardInterrupt:
        logger.warning("="*60)
        logger.warning(f"⚠️ ЗАГРУЗКА МОДЕЛИ {model_type} ПРЕРВАНА ПОЛЬЗОВАТЕЛЕМ")
        logger.warning("="*60)
        message = f"⚠️ Загрузка модели {model_type} прервана. Попробуйте скачать модель через: python download_models.py --model {model_type}"
        if show_notification:
            gr.Warning(f"Загрузка модели {model_type} прервана пользователем")
        return message
        
    except Exception as e:
        error_msg = str(e)
        logger.error("="*60)
        logger.error(f"❌ ОШИБКА ЗАГРУЗКИ МОДЕЛИ {model_type}")
        logger.error("="*60)
        logger.error(f"📝 Сообщение об ошибке: {error_msg}")
        logger.error(f"📋 Тип ошибки: {type(e).__name__}")
        logger.error(f"🔍 Детали ошибки:")
        logger.error(traceback.format_exc())
        
        # Специальные сообщения для частых проблем
        if "timeout" in error_msg.lower() or "connection" in error_msg.lower():
            message = f"❌ Ошибка сети при загрузке {model_type}. Проверьте интернет-соединение и попробуйте скачать модель через: python download_models.py --model {model_type}"
            logger.error("🌐 Проблема: Сетевое соединение")
        elif "disk" in error_msg.lower() or "space" in error_msg.lower():
            message = f"❌ Недостаточно места на диске для {model_type}. Требуется ~50GB свободного места."
            logger.error("💾 Проблема: Недостаточно места на диске")
        elif "cuda" in error_msg.lower() or "out of memory" in error_msg.lower():
            message = f"❌ Ошибка CUDA/памяти при загрузке {model_type}. Проверьте настройки в config.yaml (enable_cpu_offload: true)"
            logger.error("🎮 Проблема: CUDA/память")
        else:
            message = f"❌ Ошибка загрузки модели {model_type}: {error_msg[:100]}"
            logger.error("❓ Проблема: Неизвестная ошибка")
        
        if show_notification:
            gr.Error(f"Ошибка загрузки модели {model_type}. Проверьте логи для деталей.")
        
        return message


# Пресеты для быстрой сборки промптов
# Структура: {"Название": {"prompt": "текст промпта", "image": "путь_к_изображению_в_будущем"}}
PROMPT_PRESETS = {
    "quality": {
        "Ultra Quality": {
            "prompt": "ultra detailed, ultra high quality, masterpiece, best quality, extremely detailed",
            "image": None  # Будет добавлено в будущем
        },
        "High Quality": {
            "prompt": "high quality, detailed, sharp focus, professional",
            "image": None
        },
        "Masterpiece": {
            "prompt": "masterpiece, best quality, extremely detailed, 8k uhd",
            "image": None
        },
        "Standard": {
            "prompt": "detailed, good quality",
            "image": None
        },
    },
    "style": {
        "Реализм": {
            "prompt": "photorealistic, realistic, highly detailed, professional photography",
            "image": None
        },
        "Аниме": {
            "prompt": "anime style, manga style, japanese animation style",
            "image": None
        },
        "Фэнтези": {
            "prompt": "fantasy art, magical, mystical, ethereal",
            "image": None
        },
        "Киберпанк": {
            "prompt": "cyberpunk, neon lights, futuristic, sci-fi",
            "image": None
        },
        "Масляная живопись": {
            "prompt": "oil painting, classical art, renaissance style",
            "image": None
        },
        "Цифровая живопись": {
            "prompt": "digital art, concept art, digital painting",
            "image": None
        },
        "3D Рендер": {
            "prompt": "3d render, cgi, 3d art, rendered",
            "image": None
        },
        "Акварель": {
            "prompt": "watercolor, soft colors, gentle brushstrokes",
            "image": None
        },
        "Карандашный рисунок": {
            "prompt": "pencil sketch, graphite drawing, black and white",
            "image": None
        },
        "Пиксель-арт": {
            "prompt": "pixel art, 8-bit, retro game style",
            "image": None
        },
    },
    "lighting": {
        "Драматичное освещение": {
            "prompt": "dramatic lighting, cinematic lighting, chiaroscuro",
            "image": None
        },
        "Мягкое освещение": {
            "prompt": "soft lighting, gentle light, ambient light",
            "image": None
        },
        "Золотой час": {
            "prompt": "golden hour, warm sunlight, sunset lighting",
            "image": None
        },
        "Неоновое освещение": {
            "prompt": "neon lights, colorful lighting, vibrant glow",
            "image": None
        },
        "Естественное освещение": {
            "prompt": "natural lighting, daylight, outdoor lighting",
            "image": None
        },
        "Студийное освещение": {
            "prompt": "studio lighting, professional lighting setup",
            "image": None
        },
    },
    "composition": {
        "Крупный план": {
            "prompt": "close-up, detailed close-up, portrait",
            "image": None
        },
        "Средний план": {
            "prompt": "medium shot, full body, centered composition",
            "image": None
        },
        "Широкий план": {
            "prompt": "wide shot, landscape view, panoramic",
            "image": None
        },
        "Правило третей": {
            "prompt": "rule of thirds, balanced composition",
            "image": None
        },
        "Динамичная композиция": {
            "prompt": "dynamic composition, action pose, movement",
            "image": None
        },
    },
    "background": {
        "Без фона": {
            "prompt": "plain background, solid color background, simple background",
            "image": None
        },
        "Размытый фон": {
            "prompt": "blurred background, bokeh background, depth of field",
            "image": None
        },
        "Градиентный фон": {
            "prompt": "gradient background, colorful gradient, smooth transition",
            "image": None
        },
        "Городской пейзаж": {
            "prompt": "urban background, cityscape, modern city, skyscrapers",
            "image": None
        },
        "Природный пейзаж": {
            "prompt": "nature background, natural landscape, forest, mountains",
            "image": None
        },
        "Морской пейзаж": {
            "prompt": "ocean background, sea, beach, coastal view",
            "image": None
        },
        "Космический фон": {
            "prompt": "space background, stars, nebula, galaxy, cosmic",
            "image": None
        },
        "Интерьер": {
            "prompt": "indoor background, interior, room, indoor setting",
            "image": None
        },
        "Абстрактный фон": {
            "prompt": "abstract background, artistic background, creative background",
            "image": None
        },
        "Темный фон": {
            "prompt": "dark background, black background, shadowy background",
            "image": None
        },
        "Светлый фон": {
            "prompt": "bright background, white background, light background",
            "image": None
        },
    },
    "location": {
        "Студия": {
            "prompt": "in studio, professional studio, photography studio",
            "image": None
        },
        "Дом": {
            "prompt": "at home, indoor, home interior, cozy home",
            "image": None
        },
        "Офис": {
            "prompt": "in office, workplace, business environment, corporate setting",
            "image": None
        },
        "Кафе": {
            "prompt": "in cafe, coffee shop, restaurant, dining",
            "image": None
        },
        "Парк": {
            "prompt": "in park, public park, green space, park setting",
            "image": None
        },
        "Лес": {
            "prompt": "in forest, woodland, trees, nature",
            "image": None
        },
        "Пляж": {
            "prompt": "on beach, seaside, coastal area, beach setting",
            "image": None
        },
        "Горы": {
            "prompt": "in mountains, mountain range, alpine, high altitude",
            "image": None
        },
        "Город": {
            "prompt": "in city, urban area, downtown, city streets",
            "image": None
        },
        "Пустыня": {
            "prompt": "in desert, arid landscape, sand dunes, desert setting",
            "image": None
        },
        "Замок": {
            "prompt": "in castle, medieval castle, fortress, ancient architecture",
            "image": None
        },
        "Храм": {
            "prompt": "in temple, sacred place, religious building, ancient temple",
            "image": None
        },
        "Космос": {
            "prompt": "in space, outer space, space station, zero gravity",
            "image": None
        },
        "Под водой": {
            "prompt": "underwater, ocean depths, aquatic environment, marine setting",
            "image": None
        },
    },
    "details": {
        "Макро детали": {
            "prompt": "macro photography, extreme detail, intricate details",
            "image": None
        },
        "Текстуры": {
            "prompt": "textured surface, detailed textures, material quality",
            "image": None
        },
        "Глубина резкости": {
            "prompt": "shallow depth of field, bokeh, background blur",
            "image": None
        },
        "Резкость": {
            "prompt": "sharp focus, crisp details, high resolution",
            "image": None
        },
    }
}


def get_preset_prompt(category: str, preset_name: str) -> str:
    """Получение текста промпта из пресета"""
    if not preset_name or preset_name == "None":
        return ""
    
    preset = PROMPT_PRESETS.get(category, {}).get(preset_name)
    if preset:
        if isinstance(preset, dict):
            return preset.get("prompt", "")
        elif isinstance(preset, str):
            return preset
    return ""


def build_prompt(base_prompt: str, selected_quality: str, selected_styles: list, 
                 selected_lighting: str, selected_composition: str, 
                 selected_background: str, selected_location: str, selected_details: list,
                 additional_prompt: str = "") -> str:
    """Сборка промпта из базового описания и выбранных пресетов
    
    Args:
        base_prompt: Базовое описание от пользователя
        selected_quality: Выбранное качество
        selected_styles: Список выбранных стилей
        selected_lighting: Выбранное освещение
        selected_composition: Выбранная композиция
        selected_background: Выбранный задний план
        selected_location: Выбранная локация
        selected_details: Список выбранных деталей
        additional_prompt: Дополнительный промпт пользователя (не удаляется при обновлении)
    
    Returns:
        Собранный промпт
    """
    parts = []
    
    # Базовое описание
    if base_prompt and base_prompt.strip():
        parts.append(base_prompt.strip())
    
    # Качество
    quality_text = get_preset_prompt("quality", selected_quality)
    if quality_text:
        parts.append(quality_text)
    
    # Стили (можно выбрать несколько)
    if selected_styles:
        for style in selected_styles:
            if style and style != "None":
                style_text = get_preset_prompt("style", style)
                if style_text:
                    parts.append(style_text)
    
    # Освещение
    lighting_text = get_preset_prompt("lighting", selected_lighting)
    if lighting_text:
        parts.append(lighting_text)
    
    # Композиция
    composition_text = get_preset_prompt("composition", selected_composition)
    if composition_text:
        parts.append(composition_text)
    
    # Задний план
    background_text = get_preset_prompt("background", selected_background)
    if background_text:
        parts.append(background_text)
    
    # Место (локация)
    location_text = get_preset_prompt("location", selected_location)
    if location_text:
        parts.append(location_text)
    
    # Детали (можно выбрать несколько)
    if selected_details:
        for detail in selected_details:
            if detail and detail != "None":
                detail_text = get_preset_prompt("details", detail)
                if detail_text:
                    parts.append(detail_text)
    
    # Объединяем все части, убирая пустые строки
    final_prompt = ", ".join([p for p in parts if p and p.strip()])
    
    # Добавляем дополнительный промпт пользователя в конец (если есть)
    # Этот промпт не удаляется при автоматической сборке
    if additional_prompt and additional_prompt.strip():
        if final_prompt:
            final_prompt = f"{final_prompt}, {additional_prompt.strip()}"
        else:
            final_prompt = additional_prompt.strip()
    
    # Если промпт пустой, возвращаем подсказку
    if not final_prompt:
        return "Введите описание и выберите параметры для сборки промпта"
    
    return final_prompt


def generate_image(
    model_type: str,
    prompt: str,
    height: int,
    width: int,
    num_steps: int,
    guidance_scale: float,
    seed: int,
    progress=gr.Progress()
):
    """Функция генерации изображения"""
    global generators
    
    logger.info("="*60)
    logger.info(f"🎨 ГЕНЕРАЦИЯ ИЗОБРАЖЕНИЯ")
    logger.info("="*60)
    logger.info(f"📦 Модель: {model_type}")
    logger.info(f"📝 Промпт: {prompt[:100]}...")
    logger.info(f"📐 Размер: {width}x{height}")
    logger.info(f"🔢 Шаги: {num_steps}, Guidance: {guidance_scale}, Seed: {seed if seed != -1 else 'случайный'}")
    logger.info(f"⏰ Время начала: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Специальная обработка для ComfyUI
    if model_type == 'qwen-gguf-comfyui':
        return generate_with_comfyui(prompt, height, width, num_steps, guidance_scale, seed, progress)
    
    if model_type not in generators:
        error_msg = f"❌ Модель {model_type} не загружена. Пожалуйста, загрузите модель сначала."
        logger.error(error_msg)
        gr.Error(f"Модель {model_type} не загружена. Загрузите модель сначала.")
        return None, error_msg
    
    if not prompt or not prompt.strip():
        error_msg = "❌ Пожалуйста, введите текстовый промпт"
        logger.error(error_msg)
        gr.Warning("Введите текстовый промпт для генерации")
        return None, error_msg
    
    # Проверка доступности функции генерации
    generator = generators[model_type]
    if not hasattr(generator, 'generate') or not callable(getattr(generator, 'generate', None)):
        error_msg = f"❌ Модель {model_type} не поддерживает генерацию изображений"
        logger.error(error_msg)
        logger.error(f"🔧 Доступные функции: {[f for f in dir(generator) if not f.startswith('_')]}")
        gr.Error(f"Модель {model_type} не поддерживает генерацию")
        return None, error_msg
    
    try:
        logger.info("🚀 Начало генерации...")
        progress(0, desc="Генерация изображения...")
        
        seed_value = None if seed == -1 else seed
        
        # Для Qwen используем правильные параметры по умолчанию
        if model_type == 'qwen-image-edit':
            # Qwen требует guidance_scale = 1.0 (не 0.0 как для Turbo моделей)
            if guidance_scale == 0.0:
                guidance_scale = 1.0
                logger.info("⚠️ Автокоррекция: Использование guidance_scale=1.0 для Qwen (вместо 0.0)")
                gr.Info("Автоматически установлен guidance_scale=1.0 для Qwen")
        
        image = generator.generate(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            seed=seed_value
        )
        
        logger.info("="*60)
        logger.info("✅ ГЕНЕРАЦИЯ УСПЕШНО ЗАВЕРШЕНА")
        logger.info("="*60)
        logger.info(f"⏰ Время завершения: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"📊 Результат: Изображение {width}x{height} успешно создано")
        
        progress(1.0, desc="Готово!")
        gr.Success("Изображение успешно сгенерировано!")
        return image, "✅ Изображение успешно сгенерировано!"
        
    except Exception as e:
        error_msg = str(e)
        logger.error("="*60)
        logger.error("❌ ОШИБКА ГЕНЕРАЦИИ")
        logger.error("="*60)
        logger.error(f"📝 Сообщение об ошибке: {error_msg}")
        logger.error(f"📋 Тип ошибки: {type(e).__name__}")
        logger.error(f"🔍 Детали ошибки:")
        logger.error(traceback.format_exc())
        
        # Специальные сообщения для частых проблем
        if "cuda" in error_msg.lower() or "out of memory" in error_msg.lower():
            detailed_msg = f"❌ Ошибка CUDA/памяти: {error_msg[:100]}. Проверьте настройки в config.yaml (enable_cpu_offload: true)"
            logger.error("🎮 Проблема: CUDA/память")
        elif "timeout" in error_msg.lower():
            detailed_msg = f"❌ Таймаут при генерации: {error_msg[:100]}"
            logger.error("⏱️ Проблема: Таймаут")
        else:
            detailed_msg = f"❌ Ошибка: {error_msg[:100]}"
            logger.error("❓ Проблема: Неизвестная ошибка")
        
        gr.Error(f"Ошибка генерации. Проверьте логи для деталей.")
        return None, detailed_msg


def edit_image(
    model_type: str,
    image1,
    image2,
    prompt: str,
    negative_prompt: str,
    num_steps: int,
    guidance_scale: float,
    true_cfg_scale: float,
    seed: int,
    progress=gr.Progress()
):
    """Функция редактирования изображения"""
    global generators
    
    logger.info("="*60)
    logger.info(f"✏️ РЕДАКТИРОВАНИЕ ИЗОБРАЖЕНИЯ")
    logger.info("="*60)
    logger.info(f"📦 Модель: {model_type}")
    logger.info(f"📝 Промпт: {prompt[:100]}...")
    logger.info(f"🖼️ Изображений: {1 if image2 is None else 2}")
    logger.info(f"🔢 Шаги: {num_steps}, Guidance: {guidance_scale}, True CFG: {true_cfg_scale}, Seed: {seed if seed != -1 else 'случайный'}")
    logger.info(f"⏰ Время начала: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if model_type not in generators:
        error_msg = f"❌ Модель {model_type} не загружена. Пожалуйста, загрузите модель сначала."
        logger.error(error_msg)
        gr.Error(f"Модель {model_type} не загружена. Загрузите модель сначала.")
        return None, error_msg
    
    if image1 is None:
        error_msg = "❌ Пожалуйста, загрузите хотя бы одно изображение"
        logger.error(error_msg)
        gr.Warning("Загрузите хотя бы одно изображение для редактирования")
        return None, error_msg
    
    if not prompt or not prompt.strip():
        error_msg = "❌ Пожалуйста, введите текстовый промпт"
        logger.error(error_msg)
        gr.Warning("Введите текстовый промпт для редактирования")
        return None, error_msg
    
    # Проверка доступности функции редактирования
    generator = generators[model_type]
    if not hasattr(generator, 'edit') or not callable(getattr(generator, 'edit', None)):
        error_msg = f"❌ Модель {model_type} не поддерживает редактирование изображений"
        logger.error(error_msg)
        logger.error(f"🔧 Доступные функции: {[f for f in dir(generator) if not f.startswith('_')]}")
        gr.Error(f"Модель {model_type} не поддерживает редактирование")
        return None, error_msg
    
    try:
        logger.info("🚀 Начало редактирования...")
        progress(0, desc="Редактирование изображения...")
        
        seed_value = None if seed == -1 else seed
        
        # Подготовка списка изображений
        images = [image1]
        if image2 is not None:
            images.append(image2)
            logger.info(f"📸 Используется 2 изображения для редактирования")
        else:
            logger.info(f"📸 Используется 1 изображение для редактирования")
        
        image = generator.edit(
            images=images,
            prompt=prompt,
            negative_prompt=negative_prompt or " ",
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            true_cfg_scale=true_cfg_scale,
            seed=seed_value
        )
        
        logger.info("="*60)
        logger.info("✅ РЕДАКТИРОВАНИЕ УСПЕШНО ЗАВЕРШЕНО")
        logger.info("="*60)
        logger.info(f"⏰ Время завершения: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"📊 Результат: Изображение успешно отредактировано")
        
        progress(1.0, desc="Готово!")
        gr.Success("Изображение успешно отредактировано!")
        return image, "✅ Изображение успешно отредактировано!"
        
    except Exception as e:
        error_msg = str(e)
        logger.error("="*60)
        logger.error("❌ ОШИБКА РЕДАКТИРОВАНИЯ")
        logger.error("="*60)
        logger.error(f"📝 Сообщение об ошибке: {error_msg}")
        logger.error(f"📋 Тип ошибки: {type(e).__name__}")
        logger.error(f"🔍 Детали ошибки:")
        logger.error(traceback.format_exc())
        
        # Специальные сообщения для частых проблем
        if "cuda" in error_msg.lower() or "out of memory" in error_msg.lower():
            detailed_msg = f"❌ Ошибка CUDA/памяти: {error_msg[:100]}. Проверьте настройки в config.yaml (enable_cpu_offload: true)"
            logger.error("🎮 Проблема: CUDA/память")
        elif "timeout" in error_msg.lower():
            detailed_msg = f"❌ Таймаут при редактировании: {error_msg[:100]}"
            logger.error("⏱️ Проблема: Таймаут")
        else:
            detailed_msg = f"❌ Ошибка: {error_msg[:100]}"
            logger.error("❓ Проблема: Неизвестная ошибка")
        
        gr.Error(f"Ошибка редактирования. Проверьте логи для деталей.")
        return None, detailed_msg


def load_background_remover():
    """Загрузка модели удаления фона"""
    global background_remover
    
    if not BACKGROUND_REMOVER_AVAILABLE:
        logger.error("Модуль удаления фона недоступен")
        gr.Error("Модуль удаления фона недоступен. Установите зависимости.")
        return "❌ Модуль недоступен"
    
    if background_remover is not None:
        logger.info("RMBG-2.0 уже загружена")
        gr.Info("RMBG-2.0 уже загружена")
        return "✅ RMBG-2.0 готова к работе"
    
    try:
        logger.info("Загрузка RMBG-2.0 для удаления фона...")
        bg_config = config.get('background_removal', {})
        
        background_remover = BackgroundRemover(
            device=bg_config.get('device', 'cuda'),
            precision=bg_config.get('precision', 'fp16'),
        )
        
        if background_remover.load_model():
            logger.info("RMBG-2.0 успешно загружена")
            gr.Success("RMBG-2.0 успешно загружена!")
            return "✅ RMBG-2.0 готова к работе"
        else:
            logger.error("Не удалось загрузить RMBG-2.0")
            gr.Error("Не удалось загрузить RMBG-2.0")
            return "❌ Ошибка загрузки RMBG-2.0"
            
    except Exception as e:
        logger.error(f"Ошибка при загрузке RMBG-2.0: {e}")
        gr.Error(f"Ошибка загрузки: {str(e)[:100]}")
        return f"❌ Ошибка: {str(e)[:50]}"


def remove_background(
    image,
    progress=gr.Progress()
):
    """Удаление фона с изображения"""
    global background_remover
    
    logger.info("="*60)
    logger.info("🎭 УДАЛЕНИЕ ФОНА")
    logger.info("="*60)
    
    if image is None:
        gr.Warning("Загрузите изображение")
        return None, "❌ Изображение не загружено"
    
    if background_remover is None:
        load_result = load_background_remover()
        if "❌" in load_result:
            return None, load_result
    
    try:
        progress(0.3, desc="Удаление фона...")
        result = background_remover.remove_background(image)
        
        progress(1.0, desc="Готово!")
        logger.info("✅ Фон успешно удален")
        gr.Success("Фон успешно удален!")
        return result, "✅ Фон успешно удален!"
        
    except Exception as e:
        logger.error(f"Ошибка при удалении фона: {e}")
        gr.Error("Ошибка при удалении фона")
        return None, f"❌ Ошибка: {str(e)[:100]}"


def replace_background(
    image,
    background_type: str,
    custom_color: str,
    background_image,
    progress=gr.Progress()
):
    """Замена фона на изображении"""
    global background_remover
    
    logger.info("="*60)
    logger.info("🎨 ЗАМЕНА ФОНА")
    logger.info("="*60)
    
    if image is None:
        gr.Warning("Загрузите изображение")
        return None, "❌ Изображение не загружено"
    
    if background_remover is None:
        load_result = load_background_remover()
        if "❌" in load_result:
            return None, load_result
    
    try:
        progress(0.3, desc="Определение фона...")
        
        # Определение нового фона
        if background_type == "Изображение" and background_image is not None:
            new_background = background_image
            logger.info("Использование пользовательского изображения как фон")
        elif background_type == "Пользовательский цвет":
            # Парсинг цвета из hex
            try:
                color = custom_color.lstrip('#')
                new_background = tuple(int(color[i:i+2], 16) for i in (0, 2, 4))
                logger.info(f"Использование цвета: {new_background}")
            except:
                new_background = (255, 255, 255)
                logger.warning("Неверный формат цвета, использую белый")
        elif BACKGROUND_REMOVER_AVAILABLE and BackgroundPresets:
            # Используем пресет
            preset_name = background_type.lower().replace(' ', '_')
            preset = BackgroundPresets.get_preset(preset_name)
            if preset:
                new_background = preset
                logger.info(f"Использование пресета: {preset_name}")
            else:
                new_background = (255, 255, 255)
        else:
            new_background = (255, 255, 255)
        
        progress(0.6, desc="Замена фона...")
        result = background_remover.replace_background(image, new_background)
        
        progress(1.0, desc="Готово!")
        logger.info("✅ Фон успешно заменен")
        gr.Success("Фон успешно заменен!")
        return result, "✅ Фон успешно заменен!"
        
    except Exception as e:
        logger.error(f"Ошибка при замене фона: {e}")
        gr.Error("Ошибка при замене фона")
        return None, f"❌ Ошибка: {str(e)[:100]}"


def get_background_presets() -> list:
    """Получение списка пресетов фона"""
    presets = [
        "White Studio",
        "Black Studio", 
        "Gray Studio",
        "Blue Sky",
        "Sunset Orange",
        "Neon Pink",
        "Cyberpunk Purple",
        "Forest Green",
        "Пользовательский цвет",
        "Изображение"
    ]
    return presets


def create_interface():
    """Создание интерфейса Gradio"""
    web_config = config.get('web_interface', {})
    gen_config = config.get('generation', {})
    
    # Значения по умолчанию
    default_height = gen_config.get('default_height', 1024)
    default_width = gen_config.get('default_width', 1024)
    default_steps = gen_config.get('default_num_inference_steps', 9)
    default_guidance = gen_config.get('default_guidance_scale', 0.0)
    
    title = web_config.get('title', 'Image Generator')
    description = web_config.get('description', 'Локальный генератор изображений')
    
    available_models = get_available_models()
    
    with gr.Blocks(title=title) as app:
        gr.Markdown(f"# {title}")
        gr.Markdown(description)
        
        # Панель статуса системы
        status_display = None
        refresh_status_btn = None
        
        with gr.Accordion("📊 Статус системы и функций", open=False):
            if STATUS_CHECKER_AVAILABLE:
                status_checker = SystemStatusChecker()
                system_status = status_checker.check_system()
                
                status_html = status_checker.get_status_html()
                status_display = gr.HTML(value=status_html, label="Статус")
                
                def update_status():
                    """Обновление статуса системы"""
                    checker = SystemStatusChecker()
                    checker.check_system()
                    checker.check_functions(generators)
                    return checker.get_status_html()
                
                refresh_status_btn = gr.Button("🔄 Обновить статус", variant="secondary", size="sm")
                refresh_status_btn.click(fn=update_status, outputs=[status_display])
            else:
                gr.Markdown("⚠️ Модуль проверки статуса недоступен")
        
        # Выбор и загрузка модели
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("## 🔧 Управление моделями")
                model_selector = gr.Dropdown(
                    choices=available_models,
                    label="Выберите модель",
                    value=config.get('default_model', 'z-image-turbo'),
                    info="Выберите модель для использования"
                )
                load_model_btn = gr.Button("🔄 Загрузить модель", variant="secondary")
                model_status = gr.Textbox(
                    label="Статус модели",
                    interactive=False,
                    value="Модель не загружена"
                )
            
            # Панель информации о модели
            with gr.Column(scale=3):
                gr.Markdown("## 📋 Информация о модели")
                model_info_html = gr.HTML(
                    value="""
                    <div style="padding: 15px; background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); border-radius: 10px; color: #fff;">
                        <h3 style="margin: 0 0 10px 0; color: #00d4ff;">Z-Image-Turbo</h3>
                        <p style="margin: 5px 0; color: #aaa;">Быстрая генерация изображений (9 шагов)</p>
                        <div style="margin-top: 10px;">
                            <span style="background: #00ff88; color: #000; padding: 3px 8px; border-radius: 5px; margin-right: 5px; font-size: 12px;">✓ Генерация</span>
                            <span style="background: #ff4444; color: #fff; padding: 3px 8px; border-radius: 5px; margin-right: 5px; font-size: 12px;">✗ Редактирование</span>
                        </div>
                        <div style="margin-top: 10px; font-size: 12px; color: #888;">
                            Рекомендуемые настройки: 1024x1024, 9 шагов, guidance 0.0
                        </div>
                    </div>
                    """,
                    label="Возможности"
                )
        
        # Функция обновления информации о модели
        def update_model_info(model_type):
            info = get_model_info(model_type)
            features = info.get('features', {})
            settings = info.get('recommended_settings', {})
            
            # Создание badges для функций
            badges = []
            feature_labels = {
                'fast_generation': ('⚡ Быстрая', '#00ff88'),
                'image_editing': ('✏️ Редактирование', '#00d4ff'),
                'background_removal': ('🎭 Удаление фона', '#ff88ff'),
                'lora_support': ('🔧 LoRA', '#ffaa00'),
                'low_vram': ('💾 8GB VRAM', '#88ff88'),
            }
            
            for key, (label, color) in feature_labels.items():
                if features.get(key):
                    badges.append(f'<span style="background: {color}; color: #000; padding: 3px 8px; border-radius: 5px; margin-right: 5px; font-size: 12px;">{label}</span>')
            
            # Capabilities badges
            cap_badges = []
            if 'generate' in info.get('capabilities', []):
                cap_badges.append('<span style="background: #00ff88; color: #000; padding: 3px 8px; border-radius: 5px; margin-right: 5px; font-size: 12px;">✓ Генерация</span>')
            else:
                cap_badges.append('<span style="background: #ff4444; color: #fff; padding: 3px 8px; border-radius: 5px; margin-right: 5px; font-size: 12px;">✗ Генерация</span>')
            
            if 'edit' in info.get('capabilities', []):
                cap_badges.append('<span style="background: #00ff88; color: #000; padding: 3px 8px; border-radius: 5px; margin-right: 5px; font-size: 12px;">✓ Редактирование</span>')
            else:
                cap_badges.append('<span style="background: #ff4444; color: #fff; padding: 3px 8px; border-radius: 5px; margin-right: 5px; font-size: 12px;">✗ Редактирование</span>')
            
            # Определение цвета заголовка
            header_color = '#00d4ff' if info.get('type') == 'diffusers' else '#ff88ff'
            
            html = f"""
            <div style="padding: 15px; background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); border-radius: 10px; color: #fff;">
                <h3 style="margin: 0 0 10px 0; color: {header_color};">{info.get('name', model_type)}</h3>
                <p style="margin: 5px 0; color: #aaa;">{info.get('description', '')}</p>
                <div style="margin-top: 10px;">
                    {''.join(cap_badges)}
                </div>
                <div style="margin-top: 8px;">
                    {''.join(badges)}
                </div>
                <div style="margin-top: 10px; font-size: 12px; color: #888;">
                    Рекомендуемые настройки: {settings.get('width', 512)}x{settings.get('height', 512)}, 
                    {settings.get('steps', 20)} шагов, guidance {settings.get('guidance', 1.0)}
                </div>
                <div style="margin-top: 5px; font-size: 11px; color: #666;">
                    Тип: {info.get('type', 'unknown').upper()}
                </div>
            </div>
            """
            return html
        
        # Привязка обновления информации при смене модели
        model_selector.change(
            fn=update_model_info,
            inputs=[model_selector],
            outputs=[model_info_html]
        )
        
        # Вкладки для генерации и редактирования
        with gr.Tabs() as tabs:
            # Вкладка генерации
            with gr.Tab("🎨 Генерация изображений"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Параметры генерации")
                
                gen_model_selector = gr.Dropdown(
                    choices=available_models,
                    label="Модель для генерации",
                    value=config.get('default_model', 'z-image-turbo'),
                    info="Z-Image-Turbo (быстрая) | Qwen (качество) | ComfyUI GGUF (8GB VRAM)"
                )
                
                gr.Markdown("#### Базовое описание")
                prompt_input = gr.Textbox(
                    label="Основное описание изображения",
                    placeholder="Введите основное описание (например: 'Young Chinese woman in red Hanfu')...",
                    lines=3,
                    value=""
                )
                
                gr.Markdown("#### Быстрая сборка промпта")
                gr.Markdown("*В будущем здесь будут отображаться примеры изображений для каждого пресета*")
                
                with gr.Row():
                    quality_selector = gr.Dropdown(
                        choices=["None"] + list(PROMPT_PRESETS["quality"].keys()),
                        label="Качество",
                        value="None",
                        info="Выберите уровень качества"
                    )
                
                with gr.Row():
                    style_selector = gr.CheckboxGroup(
                        choices=list(PROMPT_PRESETS["style"].keys()),
                        label="Стили (можно выбрать несколько)",
                        info="Выберите один или несколько стилей"
                    )
                
                with gr.Row():
                    lighting_selector = gr.Dropdown(
                        choices=["None"] + list(PROMPT_PRESETS["lighting"].keys()),
                        label="Освещение",
                        value="None",
                        info="Выберите тип освещения"
                    )
                    composition_selector = gr.Dropdown(
                        choices=["None"] + list(PROMPT_PRESETS["composition"].keys()),
                        label="Композиция",
                        value="None",
                        info="Выберите композицию"
                    )
                
                with gr.Row():
                    background_selector = gr.Dropdown(
                        choices=["None"] + list(PROMPT_PRESETS["background"].keys()),
                        label="Задний план",
                        value="None",
                        info="Выберите тип заднего плана"
                    )
                    location_selector = gr.Dropdown(
                        choices=["None"] + list(PROMPT_PRESETS["location"].keys()),
                        label="Место (локация)",
                        value="None",
                        info="Выберите место действия"
                    )
                
                with gr.Row():
                    details_selector = gr.CheckboxGroup(
                        choices=list(PROMPT_PRESETS["details"].keys()),
                        label="Детали (можно выбрать несколько)",
                        info="Дополнительные детали"
                    )
                
                # Поле для отображения собранного промпта
                assembled_prompt = gr.Textbox(
                    label="Собранный промпт",
                    placeholder="Промпт будет собран автоматически...",
                    lines=4,
                    interactive=True,
                    info="Вы можете редактировать собранный промпт вручную"
                )
                
                gr.Markdown("#### Дополнительный промпт (постоянный)")
                additional_prompt = gr.Textbox(
                    label="Дополнительный промпт",
                    placeholder="Введите дополнительный текст, который будет добавлен к промпту и не будет удаляться при обновлении...",
                    lines=2,
                    info="Этот текст будет добавлен в конец промпта и сохранится при автоматической сборке"
                )
                
                # Кнопка для сборки промпта
                assemble_btn = gr.Button("🔧 Собрать промпт", variant="secondary")
                
                gr.Markdown("---")
                
                with gr.Row():
                    height_input = gr.Slider(
                        label="Высота",
                        minimum=256,  # Уменьшено для экономии памяти на 8GB VRAM
                        maximum=2048,
                        value=default_height,
                        step=64
                    )
                    width_input = gr.Slider(
                        label="Ширина",
                        minimum=256,  # Уменьшено для экономии памяти на 8GB VRAM
                        maximum=2048,
                        value=default_width,
                        step=64
                    )
                
                with gr.Row():
                    steps_input = gr.Slider(
                        label="Количество шагов",
                        minimum=1,
                        maximum=50,
                        value=default_steps,
                        step=1
                    )
                    guidance_input = gr.Slider(
                        label="Guidance Scale",
                        minimum=0.0,
                        maximum=10.0,
                        value=default_guidance,
                        step=0.1
                    )
                
                seed_input = gr.Number(
                    label="Seed (-1 для случайного)",
                    value=-1,
                    precision=0
                )
                
                generate_btn = gr.Button("🎨 Сгенерировать", variant="primary", size="lg")
                gen_status = gr.Textbox(
                    label="Статус",
                    interactive=False
                )
            
            with gr.Column(scale=1):
                gr.Markdown("### Результат")
                gen_image_output = gr.Image(
                    label="Сгенерированное изображение",
                    type="pil",
                    height=600
                )
                
                # Примеры промптов
                gr.Markdown("### Примеры промптов")
                examples = [
                    "Young Chinese woman in red Hanfu, intricate embroidery. Impeccable makeup, red floral forehead pattern.",
                    "A futuristic cityscape at sunset, neon lights, cyberpunk style, highly detailed",
                    "A serene Japanese garden with cherry blossoms, traditional architecture, peaceful atmosphere",
                    "Portrait of a wise old wizard with a long beard, magical staff, fantasy art style"
                ]
                gr.Examples(
                    examples=[[ex] for ex in examples],
                    inputs=[assembled_prompt]
                )
            
            # Вкладка редактирования
            with gr.Tab("✏️ Редактирование изображений"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Параметры редактирования")
                        
                        edit_model_selector = gr.Dropdown(
                            choices=available_models,
                            label="Модель",
                            value='qwen-image-edit',
                            info="Qwen-Image-Edit для редактирования"
                        )
                        
                        edit_prompt_input = gr.Textbox(
                            label="Текстовый промпт",
                            placeholder="Опишите желаемые изменения...",
                            lines=3,
                            value="The magician bear is on the left, the alchemist bear is on the right, facing each other in the central park square."
                        )
                        
                        negative_prompt_input = gr.Textbox(
                            label="Негативный промпт (опционально)",
                            placeholder=" ",
                            lines=2,
                            value=" "
                        )
                        
                        with gr.Row():
                            edit_steps_input = gr.Slider(
                                label="Количество шагов",
                                minimum=1,
                                maximum=100,
                                value=40,
                                step=1
                            )
                            edit_guidance_input = gr.Slider(
                                label="Guidance Scale",
                                minimum=0.0,
                                maximum=10.0,
                                value=1.0,
                                step=0.1
                            )
                        
                        true_cfg_input = gr.Slider(
                            label="True CFG Scale",
                            minimum=0.0,
                            maximum=10.0,
                            value=4.0,
                            step=0.1
                        )
                        
                        edit_seed_input = gr.Number(
                            label="Seed (-1 для случайного)",
                            value=-1,
                            precision=0
                        )
                        
                        edit_btn = gr.Button("✏️ Отредактировать", variant="primary", size="lg")
                        edit_status = gr.Textbox(
                            label="Статус",
                            interactive=False
                        )
                    
                    with gr.Column(scale=1):
                        gr.Markdown("### Входные изображения")
                        edit_image1_input = gr.Image(
                            label="Изображение 1 (обязательно)",
                            type="pil",
                            height=300
                        )
                        edit_image2_input = gr.Image(
                            label="Изображение 2 (опционально)",
                            type="pil",
                            height=300
                        )
                        
                        gr.Markdown("### Результат")
                        edit_image_output = gr.Image(
                            label="Отредактированное изображение",
                            type="pil",
                            height=600
                        )
            
            # Вкладка удаления/замены фона
            with gr.Tab("🎭 Удаление фона"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Удаление и замена фона")
                        gr.Markdown("*Использует модель RMBG-2.0 для точного удаления фона*")
                        
                        # Кнопка загрузки модели
                        load_bg_remover_btn = gr.Button("📥 Загрузить RMBG-2.0", variant="secondary")
                        bg_remover_status = gr.Textbox(
                            label="Статус модели",
                            value="Модель не загружена",
                            interactive=False
                        )
                        
                        gr.Markdown("---")
                        gr.Markdown("#### Входное изображение")
                        bg_input_image = gr.Image(
                            label="Изображение для обработки",
                            type="pil",
                            height=300
                        )
                        
                        gr.Markdown("#### Операция")
                        bg_operation = gr.Radio(
                            choices=["Удалить фон", "Заменить фон"],
                            value="Удалить фон",
                            label="Выберите операцию"
                        )
                        
                        # Настройки замены фона
                        with gr.Group(visible=False) as bg_replace_settings:
                            gr.Markdown("#### Настройки нового фона")
                            bg_preset = gr.Dropdown(
                                choices=get_background_presets(),
                                value="White Studio",
                                label="Пресет фона"
                            )
                            bg_custom_color = gr.ColorPicker(
                                value="#FFFFFF",
                                label="Пользовательский цвет",
                                visible=False
                            )
                            bg_custom_image = gr.Image(
                                label="Изображение для фона",
                                type="pil",
                                height=150,
                                visible=False
                            )
                        
                        bg_process_btn = gr.Button("🎭 Обработать", variant="primary", size="lg")
                        bg_status = gr.Textbox(
                            label="Статус",
                            interactive=False
                        )
                    
                    with gr.Column(scale=1):
                        gr.Markdown("### Результат")
                        bg_output_image = gr.Image(
                            label="Обработанное изображение",
                            type="pil",
                            height=500
                        )
                        
                        gr.Markdown("### Подсказки")
                        gr.Markdown("""
                        **Удаление фона:**
                        - Создает изображение с прозрачным фоном (PNG)
                        - Идеально для объектов и портретов
                        
                        **Замена фона:**
                        - Выберите пресет или свой цвет/изображение
                        - Автоматически удаляет старый фон и добавляет новый
                        
                        **Пресеты:**
                        - White/Black/Gray Studio - студийные фоны
                        - Blue Sky - небесно-голубой
                        - Cyberpunk Purple - неоновый фиолетовый
                        """)
                
                # Обработчики для вкладки удаления фона
                def toggle_bg_settings(operation):
                    """Переключение видимости настроек замены фона"""
                    show_replace = operation == "Заменить фон"
                    return gr.update(visible=show_replace)
                
                def toggle_custom_inputs(preset):
                    """Переключение видимости пользовательских настроек"""
                    show_color = preset == "Пользовательский цвет"
                    show_image = preset == "Изображение"
                    return gr.update(visible=show_color), gr.update(visible=show_image)
                
                def process_background(image, operation, preset, custom_color, custom_image, progress=gr.Progress()):
                    """Обработка фона"""
                    if operation == "Удалить фон":
                        return remove_background(image, progress)
                    else:
                        return replace_background(image, preset, custom_color, custom_image, progress)
                
                bg_operation.change(
                    fn=toggle_bg_settings,
                    inputs=[bg_operation],
                    outputs=[bg_replace_settings]
                )
                
                bg_preset.change(
                    fn=toggle_custom_inputs,
                    inputs=[bg_preset],
                    outputs=[bg_custom_color, bg_custom_image]
                )
                
                load_bg_remover_btn.click(
                    fn=load_background_remover,
                    outputs=[bg_remover_status]
                )
                
                bg_process_btn.click(
                    fn=process_background,
                    inputs=[bg_input_image, bg_operation, bg_preset, bg_custom_color, bg_custom_image],
                    outputs=[bg_output_image, bg_status]
                )
        
        # Функция для обновления параметров при смене модели генерации
        def update_generation_params(model_type: str):
            """Обновление параметров генерации в зависимости от модели"""
            # Получаем информацию о модели
            info = get_model_info(model_type)
            settings = info.get('recommended_settings', {})
            
            if model_type == 'qwen-image-edit':
                # Параметры для Qwen (diffusers)
                return (
                    settings.get('height', 512),
                    settings.get('width', 512),
                    settings.get('steps', 40),
                    settings.get('guidance', 1.0)
                )
            elif model_type == 'qwen-gguf-comfyui':
                # Параметры для ComfyUI GGUF (быстрая генерация)
                return (
                    settings.get('height', 512),
                    settings.get('width', 512),
                    settings.get('steps', 4),  # Lightning LoRA style
                    settings.get('guidance', 1.0)
                )
            else:
                # Параметры для Z-Image-Turbo
                return (
                    settings.get('height', default_height),
                    settings.get('width', default_width),
                    settings.get('steps', default_steps),
                    settings.get('guidance', default_guidance)
                )
        
        # Обработчики событий
        def load_model_with_status_update(model_type: str):
            """Загрузка модели с обновлением статуса"""
            result = load_model(model_type, show_notification=True)
            return result
        
        def update_status_display():
            """Обновление отображения статуса"""
            if STATUS_CHECKER_AVAILABLE and status_display is not None:
                checker = SystemStatusChecker()
                checker.check_system()
                checker.check_functions(generators)
                return checker.get_status_html()
            return None
        
        load_model_btn.click(
            fn=load_model_with_status_update,
            inputs=[model_selector],
            outputs=[model_status]
        )
        
        # Обновление панели статуса после загрузки модели
        if STATUS_CHECKER_AVAILABLE and status_display is not None:
            load_model_btn.click(
                fn=update_status_display,
                outputs=[status_display]
            )
        
        # Обновление параметров при смене модели генерации
        gen_model_selector.change(
            fn=update_generation_params,
            inputs=[gen_model_selector],
            outputs=[height_input, width_input, steps_input, guidance_input]
        )
        
        # Также обновляем информацию о модели при смене модели генерации
        gen_model_selector.change(
            fn=update_model_info,
            inputs=[gen_model_selector],
            outputs=[model_info_html]
        )
        
        # Обработчик сборки промпта
        assemble_btn.click(
            fn=build_prompt,
            inputs=[prompt_input, quality_selector, style_selector, 
                   lighting_selector, composition_selector, background_selector,
                   location_selector, details_selector, additional_prompt],
            outputs=[assembled_prompt]
        )
        
        # Автоматическая сборка при изменении компонентов (кроме additional_prompt)
        # additional_prompt имеет свой обработчик, чтобы не терять текст пользователя
        for component in [quality_selector, style_selector, lighting_selector, 
                         composition_selector, background_selector, location_selector, details_selector]:
            component.change(
                fn=build_prompt,
                inputs=[prompt_input, quality_selector, style_selector, 
                       lighting_selector, composition_selector, background_selector,
                       location_selector, details_selector, additional_prompt],
                outputs=[assembled_prompt]
            )
        
        # При изменении базового промпта тоже обновляем собранный
        prompt_input.change(
            fn=build_prompt,
            inputs=[prompt_input, quality_selector, style_selector, 
                   lighting_selector, composition_selector, background_selector,
                   location_selector, details_selector, additional_prompt],
            outputs=[assembled_prompt]
        )
        
        # При изменении дополнительного промпта обновляем собранный промпт
        # Это позволяет пользователю видеть результат сразу, сохраняя текст в поле
        additional_prompt.change(
            fn=build_prompt,
            inputs=[prompt_input, quality_selector, style_selector, 
                   lighting_selector, composition_selector, background_selector,
                   location_selector, details_selector, additional_prompt],
            outputs=[assembled_prompt]
        )
        
        generate_btn.click(
            fn=generate_image,
            inputs=[gen_model_selector, assembled_prompt, height_input, width_input, 
                   steps_input, guidance_input, seed_input],
            outputs=[gen_image_output, gen_status]
        )
        
        edit_btn.click(
            fn=edit_image,
            inputs=[edit_model_selector, edit_image1_input, edit_image2_input,
                   edit_prompt_input, negative_prompt_input, edit_steps_input,
                   edit_guidance_input, true_cfg_input, edit_seed_input],
            outputs=[edit_image_output, edit_status]
        )
        
        # Автозагрузка моделей отключена - модели загружаются только по нажатию кнопки "Загрузить модель"
        # Это позволяет пользователю контролировать, когда загружать модели
        logger.info("Приложение запущено. Модели будут загружены по требованию при нажатии кнопки 'Загрузить модель'")
    
    return app


if __name__ == "__main__":
    config = load_config()
    web_config = config.get('web_interface', {})
    port = web_config.get('port', 7860)
    share = web_config.get('share', False)
    
    app = create_interface()
    app.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=share,
        theme=gr.themes.Soft()  # Перемещено из Blocks в launch для Gradio 6.0
    )
