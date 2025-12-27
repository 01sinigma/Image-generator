"""
Веб-интерфейс для Image Generator на базе Gradio
Поддерживает Z-Image-Turbo для генерации изображений

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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
    try:
        return ModelFactory.get_available_models()
    except:
        return ['z-image-turbo']


def load_model(model_type: str):
    """Загрузка модели указанного типа"""
    global generators
    
    if model_type in generators:
        logger.info(f"Модель {model_type} уже загружена")
        return f"✅ Модель {model_type} готова к работе!"
    
    try:
        logger.info(f"Загрузка модели {model_type}...")
        generator = ZImageGenerator(model_type=model_type)
        generators[model_type] = generator
        logger.info(f"Модель {model_type} успешно загружена")
        return f"✅ Модель {model_type} готова к работе!"
    except Exception as e:
        logger.error(f"Ошибка загрузки модели {model_type}: {e}")
        return f"❌ Ошибка загрузки модели {model_type}: {str(e)}"


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
    
    if model_type not in generators:
        return None, f"❌ Модель {model_type} не загружена. Пожалуйста, загрузите модель сначала."
    
    if not prompt or not prompt.strip():
        return None, "❌ Пожалуйста, введите текстовый промпт"
    
    try:
        progress(0, desc="Генерация изображения...")
        
        generator = generators[model_type]
        seed_value = None if seed == -1 else seed
        
        image = generator.generate(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            seed=seed_value
        )
        
        progress(1.0, desc="Готово!")
        return image, "✅ Изображение успешно сгенерировано!"
        
    except Exception as e:
        logger.error(f"Ошибка генерации: {e}")
        return None, f"❌ Ошибка: {str(e)}"


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
        
        # Выбор и загрузка модели
        with gr.Row():
            with gr.Column():
                gr.Markdown("## Управление моделями")
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
        
        # Генерация изображений
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Параметры генерации")
                
                gen_model_selector = gr.Dropdown(
                    choices=available_models,
                    label="Модель",
                    value=config.get('default_model', 'z-image-turbo'),
                    info="Z-Image-Turbo для генерации"
                )
                
                prompt_input = gr.Textbox(
                    label="Текстовый промпт",
                    placeholder="Введите описание изображения...",
                    lines=4,
                    value="Young Chinese woman in red Hanfu, intricate embroidery. Impeccable makeup, red floral forehead pattern."
                )
                
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
            inputs=prompt_input
        )
        
        # Обработчики событий
        load_model_btn.click(
            fn=load_model,
            inputs=[model_selector],
            outputs=[model_status]
        )
        
        generate_btn.click(
            fn=generate_image,
            inputs=[gen_model_selector, prompt_input, height_input, width_input, 
                   steps_input, guidance_input, seed_input],
            outputs=[gen_image_output, gen_status]
        )
        
        # Автозагрузка модели по умолчанию при старте
        app.load(
            fn=lambda: load_model(config.get('default_model', 'z-image-turbo')),
            outputs=[model_status]
        )
    
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
