"""
CLI интерфейс для Image Generator
Поддерживает генерацию изображений
"""

import argparse
import sys
from pathlib import Path
from PIL import Image
from generator import ZImageGenerator
from models import ModelFactory
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Основная функция CLI"""
    parser = argparse.ArgumentParser(
        description="Image Generator - Локальный генератор и редактор изображений",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:

ГЕНЕРАЦИЯ ИЗОБРАЖЕНИЙ:
  # Базовая генерация
  python cli.py "A beautiful sunset over mountains"

  # С указанием модели, размеров и seed
  python cli.py "A cat wearing sunglasses" --model z-image-turbo --width 1024 --height 1024 --seed 42

  # Пакетная генерация
  python cli.py "Prompt 1" "Prompt 2" "Prompt 3" --batch
        """
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Тип модели: z-image-turbo. По умолчанию из конфигурации'
    )
    
    parser.add_argument(
        'prompts',
        nargs='*',
        help='Текстовые промпты (обязательны для генерации, опциональны для редактирования)'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Путь к файлу конфигурации (по умолчанию: config.yaml)'
    )
    
    parser.add_argument(
        '--width',
        type=int,
        default=None,
        help='Ширина изображения (по умолчанию из конфигурации)'
    )
    
    parser.add_argument(
        '--height',
        type=int,
        default=None,
        help='Высота изображения (по умолчанию из конфигурации)'
    )
    
    parser.add_argument(
        '--steps',
        type=int,
        default=None,
        help='Количество шагов инференса (по умолчанию из конфигурации)'
    )
    
    parser.add_argument(
        '--guidance-scale',
        type=float,
        default=None,
        help='Масштаб guidance (по умолчанию из конфигурации)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Случайное зерно для воспроизводимости (по умолчанию случайное)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Директория для сохранения изображений (по умолчанию из конфигурации)'
    )
    
    parser.add_argument(
        '--output-name',
        type=str,
        default=None,
        help='Имя файла для сохранения (только для одного промпта)'
    )
    
    parser.add_argument(
        '--batch',
        action='store_true',
        help='Пакетная генерация нескольких изображений'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Подробный вывод логов'
    )
    
    args = parser.parse_args()
    
    # Настройка уровня логирования
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Валидация аргументов
    if not args.prompts:
        parser.error("Требуется текстовый промпт для генерации")
    
    try:
        # Определение типа модели
        if args.model:
            model_type = args.model
        else:
            # Загрузка конфигурации для определения модели по умолчанию
            import yaml
            with open(args.config, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            model_type = config.get('default_model', 'z-image-turbo')
        
        # Инициализация генератора
        logger.info(f"Инициализация генератора с моделью: {model_type}...")
        generator = ZImageGenerator(config_path=args.config, model_type=model_type)
        logger.info("Генератор успешно инициализирован")
        
        # Подготовка параметров генерации
        gen_kwargs = {}
        if args.width is not None:
            gen_kwargs['width'] = args.width
        if args.height is not None:
            gen_kwargs['height'] = args.height
        if args.steps is not None:
            gen_kwargs['num_inference_steps'] = args.steps
        if args.guidance_scale is not None:
            gen_kwargs['guidance_scale'] = args.guidance_scale
        if args.seed is not None:
            gen_kwargs['seed'] = args.seed
        
        # Определение директории вывода
        output_dir = args.output_dir
        if output_dir is None:
            output_dir = generator.config.get('output', {}).get('directory', 'outputs')
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Генерация изображений
            if len(args.prompts) == 1 and not args.batch:
                # Одна генерация
                prompt = args.prompts[0]
                logger.info(f"Генерация изображения для промпта: {prompt[:50]}...")
                
                if args.output_name:
                    save_path = Path(output_dir) / args.output_name
                else:
                    save_path = Path(output_dir) / "generated.png"
                
                image = generator.generate(
                    prompt=prompt,
                    save_path=str(save_path),
                    **gen_kwargs
                )
                
                logger.info(f"✅ Изображение сохранено: {save_path}")
                
            else:
                # Пакетная генерация
                logger.info(f"Пакетная генерация {len(args.prompts)} изображений...")
                
                for i, prompt in enumerate(args.prompts, 1):
                    logger.info(f"[{i}/{len(args.prompts)}] Генерация: {prompt[:50]}...")
                    
                    save_path = Path(output_dir) / f"generated_{i}.png"
                    image = generator.generate(
                        prompt=prompt,
                        save_path=str(save_path),
                        **gen_kwargs
                    )
                    
                    logger.info(f"✅ [{i}/{len(args.prompts)}] Сохранено: {save_path}")
        
        logger.info("Операция завершена успешно!")
        
    except KeyboardInterrupt:
        logger.info("\nПрервано пользователем")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Ошибка: {e}", exc_info=args.verbose)
        sys.exit(1)


if __name__ == "__main__":
    main()

