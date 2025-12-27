"""
🎨 Image Generator Pro - Полнофункциональный генератор и редактор изображений
Поддержка множества моделей с уникальными UI для каждой

Модели:
- Z-Image-Turbo: Быстрая генерация
- SDXL Turbo: Универсальная + LoRA
- Pony Diffusion: Аниме/NSFW + LoRA  
- RealVisXL: Фотореализм + LoRA
- InstructPix2Pix: Редактирование по тексту
- SDXL Inpainting: Замена частей (маска)
- Qwen-Image-Edit: Мощное редактирование
"""

import gradio as gr
import yaml
import torch
import os
from pathlib import Path
from PIL import Image
from typing import Optional, Tuple, List
import logging
from datetime import datetime

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Импорт моделей
from models import ModelFactory

# ============================================
# КОНФИГУРАЦИЯ МОДЕЛЕЙ
# ============================================

MODELS_CONFIG = {
    "z-image-turbo": {
        "name": "⚡ Z-Image-Turbo",
        "category": "generation",
        "description": "Быстрая генерация изображений за 5-10 секунд",
        "speed": "⚡ Очень быстро",
        "vram": "6 GB",
        "features": ["Быстрая генерация", "9 шагов", "Высокое разрешение"],
        "settings": {
            "height": {"default": 1024, "min": 256, "max": 2048, "step": 64},
            "width": {"default": 1024, "min": 256, "max": 2048, "step": 64},
            "steps": {"default": 9, "min": 1, "max": 20, "step": 1},
            "guidance": {"default": 0.0, "min": 0.0, "max": 5.0, "step": 0.1},
        },
        "tips": [
            "💡 Используйте guidance_scale = 0 для лучших результатов",
            "💡 9 шагов оптимально для этой модели",
            "💡 Поддерживает разрешения до 2048x2048",
        ],
        "examples": [
            "A beautiful sunset over mountains, photorealistic",
            "Portrait of a woman with blue eyes, professional photo",
            "Futuristic city at night, cyberpunk style",
        ],
        "lora_support": False,
        "edit_support": False,
        "inpaint_support": False,
    },
    
    "sdxl-turbo": {
        "name": "🎨 SDXL Turbo",
        "category": "generation",
        "description": "Универсальная модель с поддержкой LoRA и без цензуры",
        "speed": "⚡ Очень быстро",
        "vram": "6 GB",
        "features": ["LoRA поддержка", "Без цензуры", "4 шага", "NSFW"],
        "settings": {
            "height": {"default": 512, "min": 256, "max": 1024, "step": 64},
            "width": {"default": 512, "min": 256, "max": 1024, "step": 64},
            "steps": {"default": 4, "min": 1, "max": 10, "step": 1},
            "guidance": {"default": 0.0, "min": 0.0, "max": 2.0, "step": 0.1},
        },
        "tips": [
            "💡 Всего 4 шага для быстрой генерации",
            "💡 Скачайте LoRA с Civitai для новых стилей",
            "💡 Цензура отключена - можно генерировать NSFW",
        ],
        "examples": [
            "Anime girl with pink hair, detailed",
            "Realistic photo of a sports car",
            "Fantasy landscape with dragons",
        ],
        "lora_support": True,
        "edit_support": False,
        "inpaint_support": False,
    },
    
    "pony-diffusion": {
        "name": "🦄 Pony Diffusion V6",
        "category": "generation",
        "description": "Специализация на аниме и NSFW контенте",
        "speed": "🔄 Средне",
        "vram": "7 GB",
        "features": ["Аниме стиль", "NSFW из коробки", "LoRA", "Высокое качество"],
        "settings": {
            "height": {"default": 1024, "min": 512, "max": 1536, "step": 64},
            "width": {"default": 1024, "min": 512, "max": 1536, "step": 64},
            "steps": {"default": 25, "min": 10, "max": 50, "step": 1},
            "guidance": {"default": 7.0, "min": 1.0, "max": 15.0, "step": 0.5},
        },
        "tips": [
            "💡 Используйте теги в стиле Danbooru: 1girl, blue_eyes, etc",
            "💡 Добавьте 'score_9, score_8_up' для лучшего качества",
            "💡 Негативный промпт: 'worst quality, low quality'",
        ],
        "examples": [
            "score_9, 1girl, blue eyes, long hair, school uniform, detailed",
            "score_9, fantasy landscape, castle, mountains, sunset",
            "score_9, 1boy, muscular, armor, sword, epic",
        ],
        "lora_support": True,
        "edit_support": False,
        "inpaint_support": False,
        "negative_prompt_default": "worst quality, low quality, blurry, bad anatomy",
    },
    
    "realvis-xl": {
        "name": "📷 RealVisXL V4",
        "category": "generation",
        "description": "Фотореалистичные изображения высокого качества",
        "speed": "🔄 Средне",
        "vram": "7 GB",
        "features": ["Фотореализм", "LoRA", "Портреты", "Пейзажи"],
        "settings": {
            "height": {"default": 1024, "min": 512, "max": 1536, "step": 64},
            "width": {"default": 1024, "min": 512, "max": 1536, "step": 64},
            "steps": {"default": 25, "min": 10, "max": 50, "step": 1},
            "guidance": {"default": 5.0, "min": 1.0, "max": 15.0, "step": 0.5},
        },
        "tips": [
            "💡 Описывайте детально: освещение, камеру, стиль",
            "💡 Добавьте 'RAW photo, 8k uhd, dslr' для реализма",
            "💡 Используйте конкретные слова: 'bokeh', 'soft lighting'",
        ],
        "examples": [
            "RAW photo, portrait of a woman, natural lighting, 8k uhd",
            "Professional photo of a modern kitchen interior, magazine quality",
            "Landscape photo, mountains at golden hour, National Geographic",
        ],
        "lora_support": True,
        "edit_support": False,
        "inpaint_support": False,
        "negative_prompt_default": "cartoon, anime, drawing, painting, blurry, low quality",
    },
    
    "instruct-pix2pix": {
        "name": "🖌️ InstructPix2Pix",
        "category": "editing",
        "description": "Редактирование изображений по текстовой инструкции",
        "speed": "⚡ Быстро",
        "vram": "5 GB",
        "features": ["Редактирование по тексту", "Быстро", "Простые команды"],
        "settings": {
            "steps": {"default": 20, "min": 10, "max": 50, "step": 1},
            "guidance": {"default": 7.5, "min": 1.0, "max": 15.0, "step": 0.5},
            "image_guidance": {"default": 1.5, "min": 1.0, "max": 3.0, "step": 0.1},
        },
        "tips": [
            "💡 Используйте простые команды: 'сделай волосы синими'",
            "💡 Image Guidance: выше = меньше изменений",
            "💡 Можно комбинировать: 'добавь очки и бороду'",
        ],
        "examples": [
            "make the hair blue",
            "add sunglasses",
            "turn it into a painting",
            "make it night time",
            "add a beard",
            "make the person smile",
        ],
        "lora_support": False,
        "edit_support": True,
        "inpaint_support": False,
        "requires_image": True,
    },
    
    "sdxl-inpainting": {
        "name": "🎭 SDXL Inpainting",
        "category": "inpainting",
        "description": "Замена или удаление частей изображения по маске",
        "speed": "🔄 Средне",
        "vram": "7 GB",
        "features": ["Маска для замены", "LoRA", "Удаление объектов", "Замена фона"],
        "settings": {
            "steps": {"default": 25, "min": 10, "max": 50, "step": 1},
            "guidance": {"default": 7.5, "min": 1.0, "max": 15.0, "step": 0.5},
            "strength": {"default": 0.99, "min": 0.5, "max": 1.0, "step": 0.01},
        },
        "tips": [
            "💡 Белое на маске = будет заменено",
            "💡 Чёрное на маске = останется как есть",
            "💡 Strength: выше = больше изменений",
        ],
        "examples": [
            "beautiful sunset sky with clouds",
            "empty room, clean floor",
            "green forest background",
        ],
        "lora_support": True,
        "edit_support": False,
        "inpaint_support": True,
        "requires_image": True,
        "requires_mask": True,
    },
    
    "qwen-image-edit": {
        "name": "✨ Qwen-Image-Edit",
        "category": "editing",
        "description": "Мощное редактирование с пониманием контекста (медленно)",
        "speed": "🐢 Медленно (1-3 мин)",
        "vram": "4-6 GB + 20 GB RAM",
        "features": ["Мощное редактирование", "Понимание контекста", "Объединение фото", "Изменение позы"],
        "settings": {
            "height": {"default": 512, "min": 256, "max": 1024, "step": 64},
            "width": {"default": 512, "min": 256, "max": 1024, "step": 64},
            "steps": {"default": 40, "min": 20, "max": 80, "step": 1},
            "guidance": {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.1},
            "true_cfg": {"default": 4.0, "min": 1.0, "max": 10.0, "step": 0.5},
        },
        "tips": [
            "💡 Можно загрузить до 2 изображений",
            "💡 Опишите что сделать с изображениями",
            "💡 Требует 32GB RAM для работы",
            "💡 Генерация занимает 1-3 минуты",
        ],
        "examples": [
            "Put the person from image 1 into the scene from image 2",
            "Change the hairstyle to long curly hair",
            "Make the person wear a red dress",
        ],
        "lora_support": False,
        "edit_support": True,
        "inpaint_support": False,
        "requires_image": True,
        "multi_image": True,
    },
}

# Глобальные переменные
current_model = None
current_model_name = None

def load_config():
    """Загрузка конфигурации"""
    try:
        with open("config.yaml", 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except:
        return {}

config = load_config()

# ============================================
# ФУНКЦИИ РАБОТЫ С МОДЕЛЯМИ
# ============================================

def get_model_list():
    """Получение списка моделей для dropdown"""
    models = []
    for key, info in MODELS_CONFIG.items():
        models.append(f"{info['name']} - {info['description'][:40]}...")
    return models

def get_model_key_from_display(display_name: str) -> str:
    """Получение ключа модели из отображаемого имени"""
    for key, info in MODELS_CONFIG.items():
        if display_name.startswith(info['name']):
            return key
    return "z-image-turbo"

def load_model(model_display_name: str):
    """Загрузка выбранной модели"""
    global current_model, current_model_name
    
    model_key = get_model_key_from_display(model_display_name)
    model_info = MODELS_CONFIG.get(model_key, {})
    
    gr.Info(f"⏳ Загрузка модели {model_info.get('name', model_key)}...")
    logger.info(f"Загрузка модели: {model_key}")
    
    try:
        # Выгружаем предыдущую модель
        if current_model is not None:
            del current_model
            current_model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Загружаем новую
        current_model = ModelFactory.create_generator(model_key)
        current_model.load_model()
        current_model_name = model_key
        
        gr.Info(f"✅ Модель {model_info.get('name', model_key)} загружена!")
        return f"✅ Модель загружена: {model_info.get('name', model_key)}"
        
    except Exception as e:
        logger.error(f"Ошибка загрузки модели: {e}")
        gr.Error(f"❌ Ошибка: {str(e)}")
        return f"❌ Ошибка загрузки: {str(e)}"

def get_model_info_html(model_display_name: str) -> str:
    """Генерация HTML с информацией о модели"""
    model_key = get_model_key_from_display(model_display_name)
    info = MODELS_CONFIG.get(model_key, {})
    
    if not info:
        return "<p>Выберите модель</p>"
    
    features_html = "".join([f"<span class='feature-tag'>{f}</span>" for f in info.get('features', [])])
    tips_html = "".join([f"<li>{tip}</li>" for tip in info.get('tips', [])])
    examples_html = "".join([f"<code>{ex}</code><br>" for ex in info.get('examples', [])[:3]])
    
    return f"""
    <div class="model-info-card">
        <h3>{info.get('name', 'Unknown')}</h3>
        <p class="description">{info.get('description', '')}</p>
        
        <div class="stats">
            <span class="stat">⚡ {info.get('speed', 'N/A')}</span>
            <span class="stat">💾 {info.get('vram', 'N/A')} VRAM</span>
            <span class="stat">📁 {info.get('category', 'N/A').upper()}</span>
        </div>
        
        <div class="features">
            {features_html}
        </div>
        
        <details>
            <summary>💡 Советы по использованию</summary>
            <ul>{tips_html}</ul>
        </details>
        
        <details>
            <summary>📝 Примеры промптов</summary>
            <div class="examples">{examples_html}</div>
        </details>
    </div>
    """

def update_ui_for_model(model_display_name: str):
    """Обновление UI при смене модели"""
    model_key = get_model_key_from_display(model_display_name)
    info = MODELS_CONFIG.get(model_key, {})
    settings = info.get('settings', {})
    
    # Получаем значения по умолчанию
    height = settings.get('height', {}).get('default', 512)
    width = settings.get('width', {}).get('default', 512)
    steps = settings.get('steps', {}).get('default', 20)
    guidance = settings.get('guidance', {}).get('default', 7.0)
    
    # Определяем видимость элементов
    requires_image = info.get('requires_image', False)
    requires_mask = info.get('requires_mask', False)
    is_edit_mode = info.get('edit_support', False)
    is_inpaint_mode = info.get('inpaint_support', False)
    has_lora = info.get('lora_support', False)
    
    # HTML с информацией
    info_html = get_model_info_html(model_display_name)
    
    # Негативный промпт по умолчанию
    neg_prompt = info.get('negative_prompt_default', '')
    
    return (
        info_html,  # model_info
        height,     # height slider
        width,      # width slider
        steps,      # steps slider
        guidance,   # guidance slider
        gr.update(visible=requires_image),   # input_image
        gr.update(visible=requires_mask),    # mask_image
        gr.update(visible=has_lora),         # lora_section
        gr.update(visible=is_edit_mode or is_inpaint_mode),  # edit_section
        neg_prompt,  # negative prompt
    )

# ============================================
# ФУНКЦИИ ГЕНЕРАЦИИ/РЕДАКТИРОВАНИЯ
# ============================================

def generate_image(
    prompt: str,
    negative_prompt: str,
    height: int,
    width: int,
    steps: int,
    guidance: float,
    seed: int,
    input_image: Optional[Image.Image] = None,
    mask_image: Optional[Image.Image] = None,
    image_guidance: float = 1.5,
    true_cfg: float = 4.0,
    strength: float = 0.99,
):
    """Универсальная функция генерации/редактирования"""
    global current_model, current_model_name
    
    if current_model is None:
        gr.Warning("⚠️ Сначала загрузите модель!")
        return None, "❌ Модель не загружена"
    
    model_info = MODELS_CONFIG.get(current_model_name, {})
    
    # Проверка входных данных
    if model_info.get('requires_image') and input_image is None:
        gr.Warning("⚠️ Эта модель требует входное изображение!")
        return None, "❌ Загрузите изображение"
    
    if model_info.get('requires_mask') and mask_image is None:
        gr.Warning("⚠️ Эта модель требует маску!")
        return None, "❌ Нарисуйте маску"
    
    # Генерация seed если не указан
    if seed == -1:
        seed = torch.randint(0, 2**32 - 1, (1,)).item()
    
    try:
        gr.Info(f"🎨 Генерация... ({steps} шагов)")
        start_time = datetime.now()
        
        # Определяем тип операции
        if model_info.get('inpaint_support') and mask_image is not None:
            # Inpainting
            result = current_model.inpaint(
                image=input_image,
                mask=mask_image,
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=steps,
                guidance_scale=guidance,
                strength=strength,
                seed=seed,
            )
        elif model_info.get('edit_support') and input_image is not None:
            # Редактирование
            if current_model_name == "instruct-pix2pix":
                result = current_model.edit(
                    image=input_image,
                    prompt=prompt,
                    num_inference_steps=steps,
                    guidance_scale=guidance,
                    image_guidance_scale=image_guidance,
                    seed=seed,
                )
            elif current_model_name == "qwen-image-edit":
                result = current_model.edit(
                    images=[input_image],
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=steps,
                    guidance_scale=guidance,
                    true_cfg_scale=true_cfg,
                    seed=seed,
                )
            else:
                result = current_model.edit(
                    image=input_image,
                    prompt=prompt,
                    seed=seed,
                )
        else:
            # Генерация с нуля
            result = current_model.generate(
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_inference_steps=steps,
                guidance_scale=guidance,
                seed=seed,
            )
        
        elapsed = (datetime.now() - start_time).total_seconds()
        
        # Сохранение
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{current_model_name}_{timestamp}.png"
        filepath = output_dir / filename
        result.save(filepath)
        
        status = f"✅ Готово за {elapsed:.1f} сек | Seed: {seed} | Сохранено: {filename}"
        gr.Info(status)
        
        return result, status
        
    except Exception as e:
        logger.error(f"Ошибка генерации: {e}")
        gr.Error(f"❌ Ошибка: {str(e)}")
        return None, f"❌ Ошибка: {str(e)}"

# ============================================
# СОЗДАНИЕ ИНТЕРФЕЙСА
# ============================================

def create_interface():
    """Создание главного интерфейса"""
    
    # CSS стили
    css = """
    .model-info-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        border: 1px solid #0f3460;
    }
    .model-info-card h3 {
        color: #e94560;
        margin: 0 0 10px 0;
        font-size: 1.5em;
    }
    .model-info-card .description {
        color: #a0a0a0;
        margin-bottom: 15px;
    }
    .model-info-card .stats {
        display: flex;
        gap: 15px;
        margin-bottom: 15px;
        flex-wrap: wrap;
    }
    .model-info-card .stat {
        background: #0f3460;
        padding: 5px 12px;
        border-radius: 20px;
        font-size: 0.9em;
        color: #fff;
    }
    .model-info-card .features {
        margin-bottom: 15px;
    }
    .feature-tag {
        display: inline-block;
        background: #e94560;
        color: white;
        padding: 3px 10px;
        border-radius: 12px;
        margin: 2px;
        font-size: 0.8em;
    }
    .model-info-card details {
        margin-top: 10px;
        color: #ccc;
    }
    .model-info-card summary {
        cursor: pointer;
        color: #e94560;
        font-weight: bold;
    }
    .model-info-card .examples code {
        display: block;
        background: #0a0a15;
        padding: 8px;
        margin: 5px 0;
        border-radius: 5px;
        font-size: 0.85em;
        color: #00ff88;
    }
    .gradio-container {
        max-width: 1400px !important;
    }
    """
    
    with gr.Blocks(css=css, title="🎨 Image Generator Pro", theme=gr.themes.Soft(
        primary_hue="pink",
        secondary_hue="blue",
    )) as demo:
        
        gr.Markdown("""
        # 🎨 Image Generator Pro
        ### Генерация и редактирование изображений с AI
        """)
        
        with gr.Row():
            # ===== ЛЕВАЯ КОЛОНКА - Настройки =====
            with gr.Column(scale=1):
                
                # Выбор модели
                gr.Markdown("### 🤖 Выбор модели")
                model_dropdown = gr.Dropdown(
                    choices=get_model_list(),
                    value=get_model_list()[0],
                    label="Модель",
                    interactive=True,
                )
                
                load_btn = gr.Button("📥 Загрузить модель", variant="primary", size="lg")
                load_status = gr.Textbox(label="Статус", interactive=False, lines=1)
                
                # Информация о модели
                model_info = gr.HTML(get_model_info_html(get_model_list()[0]))
                
                # Секция LoRA (скрыта по умолчанию)
                with gr.Group(visible=False) as lora_section:
                    gr.Markdown("### 🎭 LoRA")
                    lora_file = gr.File(label="Загрузить LoRA (.safetensors)", file_types=[".safetensors"])
                    lora_weight = gr.Slider(0.0, 1.5, value=0.8, step=0.05, label="Вес LoRA")
                
            # ===== ЦЕНТРАЛЬНАЯ КОЛОНКА - Ввод =====
            with gr.Column(scale=2):
                
                gr.Markdown("### ✍️ Промпт")
                prompt = gr.Textbox(
                    label="Что сгенерировать?",
                    placeholder="Опишите изображение...",
                    lines=3,
                )
                
                negative_prompt = gr.Textbox(
                    label="Негативный промпт (чего избегать)",
                    placeholder="blurry, low quality, bad anatomy...",
                    lines=2,
                )
                
                # Входные изображения (для редактирования)
                with gr.Group(visible=False) as edit_section:
                    gr.Markdown("### 🖼️ Входное изображение")
                    with gr.Row():
                        input_image = gr.Image(
                            label="Изображение для редактирования",
                            type="pil",
                            visible=False,
                        )
                        mask_image = gr.Image(
                            label="Маска (белое = заменить)",
                            type="pil",
                            visible=False,
                            tool="sketch",
                        )
                
                # Параметры генерации
                gr.Markdown("### ⚙️ Параметры")
                with gr.Row():
                    height = gr.Slider(256, 2048, value=1024, step=64, label="Высота")
                    width = gr.Slider(256, 2048, value=1024, step=64, label="Ширина")
                
                with gr.Row():
                    steps = gr.Slider(1, 80, value=20, step=1, label="Шаги")
                    guidance = gr.Slider(0.0, 20.0, value=7.0, step=0.1, label="Guidance Scale")
                
                with gr.Row():
                    seed = gr.Number(value=-1, label="Seed (-1 = случайный)", precision=0)
                    image_guidance = gr.Slider(1.0, 3.0, value=1.5, step=0.1, 
                                               label="Image Guidance (для редактирования)",
                                               visible=False)
                
                with gr.Row():
                    true_cfg = gr.Slider(1.0, 10.0, value=4.0, step=0.5,
                                        label="True CFG (для Qwen)",
                                        visible=False)
                    strength = gr.Slider(0.5, 1.0, value=0.99, step=0.01,
                                        label="Strength (для Inpainting)",
                                        visible=False)
                
                # Кнопка генерации
                generate_btn = gr.Button("🎨 Сгенерировать", variant="primary", size="lg")
            
            # ===== ПРАВАЯ КОЛОНКА - Результат =====
            with gr.Column(scale=2):
                gr.Markdown("### 🖼️ Результат")
                output_image = gr.Image(label="Сгенерированное изображение", type="pil")
                output_status = gr.Textbox(label="Информация", interactive=False)
                
                with gr.Row():
                    save_btn = gr.Button("💾 Сохранить")
                    copy_seed_btn = gr.Button("🎲 Копировать Seed")
        
        # ===== СОБЫТИЯ =====
        
        # Загрузка модели
        load_btn.click(
            fn=load_model,
            inputs=[model_dropdown],
            outputs=[load_status],
        )
        
        # Обновление UI при смене модели
        model_dropdown.change(
            fn=update_ui_for_model,
            inputs=[model_dropdown],
            outputs=[
                model_info,
                height,
                width,
                steps,
                guidance,
                input_image,
                mask_image,
                lora_section,
                edit_section,
                negative_prompt,
            ],
        )
        
        # Генерация
        generate_btn.click(
            fn=generate_image,
            inputs=[
                prompt,
                negative_prompt,
                height,
                width,
                steps,
                guidance,
                seed,
                input_image,
                mask_image,
                image_guidance,
                true_cfg,
                strength,
            ],
            outputs=[output_image, output_status],
        )
        
    return demo

# ============================================
# ЗАПУСК
# ============================================

if __name__ == "__main__":
    print("=" * 60)
    print("🎨 IMAGE GENERATOR PRO")
    print("=" * 60)
    print()
    print("Доступные модели:")
    for key, info in MODELS_CONFIG.items():
        print(f"  {info['name']}: {info['description']}")
    print()
    print("=" * 60)
    
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )

