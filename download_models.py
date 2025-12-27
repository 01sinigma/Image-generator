"""
Скрипт для скачивания моделей локально
Скачивает Z-Image-Turbo и Qwen-Image-Edit-2511 в локальную директорию
"""

import os
import sys
from pathlib import Path
from huggingface_hub import snapshot_download
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def download_model(model_id: str, local_dir: str = None, resume_download: bool = True):
    """
    Скачивание модели из Hugging Face
    
    Args:
        model_id: ID модели на Hugging Face (например, "Tongyi-MAI/Z-Image-Turbo")
        local_dir: Локальная директория для сохранения (по умолчанию ./models/{model_name})
        resume_download: Продолжить загрузку если прервана
    """
    if local_dir is None:
        # Создаем директорию на основе имени модели
        model_name = model_id.split('/')[-1]
        local_dir = f"./models/{model_name}"
    
    logger.info(f"Начинаю загрузку модели: {model_id}")
    logger.info(f"Директория сохранения: {os.path.abspath(local_dir)}")
    
    try:
        # Скачивание модели
        snapshot_download(
            repo_id=model_id,
            local_dir=local_dir,
            resume_download=resume_download,
            local_dir_use_symlinks=False  # Копируем файлы, а не создаем симлинки
        )
        
        logger.info(f"✅ Модель {model_id} успешно загружена в {local_dir}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Ошибка при загрузке модели {model_id}: {e}")
        return False


def main():
    """Основная функция"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Скачивание моделей для Image Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:

  # Скачать все модели
  python download_models.py --all

  # Скачать только Z-Image-Turbo
  python download_models.py --model z-image-turbo

  # Скачать только Qwen-Image-Edit
  python download_models.py --model qwen-image-edit

  # Скачать в конкретную директорию
  python download_models.py --model z-image-turbo --output-dir ./my_models
        """
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='Скачать все модели'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        choices=['z-image-turbo', 'qwen-image-edit', 'all'],
        help='Модель для скачивания: z-image-turbo, qwen-image-edit, или all'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Базовая директория для сохранения моделей (по умолчанию ./models)'
    )
    
    parser.add_argument(
        '--resume',
        action='store_true',
        default=True,
        help='Продолжить загрузку если прервана (по умолчанию включено)'
    )
    
    parser.add_argument(
        '--no-resume',
        action='store_false',
        dest='resume',
        help='Не продолжать загрузку, начать заново'
    )
    
    args = parser.parse_args()
    
    # Определение моделей для скачивания
    models_to_download = {}
    
    if args.all or args.model == 'all':
        models_to_download = {
            'z-image-turbo': 'Tongyi-MAI/Z-Image-Turbo',
            'qwen-image-edit': 'Qwen/Qwen-Image-Edit-2511'
        }
    elif args.model == 'z-image-turbo':
        models_to_download = {
            'z-image-turbo': 'Tongyi-MAI/Z-Image-Turbo'
        }
    elif args.model == 'qwen-image-edit':
        models_to_download = {
            'qwen-image-edit': 'Qwen/Qwen-Image-Edit-2511'
        }
    else:
        # По умолчанию скачиваем все
        models_to_download = {
            'z-image-turbo': 'Tongyi-MAI/Z-Image-Turbo',
            'qwen-image-edit': 'Qwen/Qwen-Image-Edit-2511'
        }
    
    if not models_to_download:
        logger.error("Не указаны модели для скачивания")
        sys.exit(1)
    
    # Базовая директория
    base_dir = args.output_dir or "./models"
    Path(base_dir).mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("Начинаю загрузку моделей")
    logger.info(f"Всего моделей: {len(models_to_download)}")
    logger.info("=" * 60)
    
    # Скачивание моделей
    results = {}
    for model_name, model_id in models_to_download.items():
        logger.info("")
        logger.info(f"[{model_name}] Загрузка {model_id}...")
        
        if args.output_dir:
            local_dir = os.path.join(args.output_dir, model_name)
        else:
            local_dir = os.path.join(base_dir, model_name)
        
        success = download_model(model_id, local_dir, args.resume)
        results[model_name] = success
        
        if success:
            # Показываем размер директории
            try:
                total_size = sum(
                    f.stat().st_size for f in Path(local_dir).rglob('*') if f.is_file()
                )
                size_gb = total_size / (1024 ** 3)
                logger.info(f"[{model_name}] Размер: {size_gb:.2f} GB")
            except:
                pass
    
    # Итоги
    logger.info("")
    logger.info("=" * 60)
    logger.info("Итоги загрузки:")
    logger.info("=" * 60)
    
    for model_name, success in results.items():
        status = "✅ Успешно" if success else "❌ Ошибка"
        logger.info(f"{model_name}: {status}")
    
    # Проверка успешности
    all_success = all(results.values())
    
    if all_success:
        logger.info("")
        logger.info("🎉 Все модели успешно загружены!")
        logger.info("")
        logger.info("Теперь вы можете использовать модели локально.")
        logger.info("Убедитесь, что в config.yaml указаны правильные пути к моделям.")
    else:
        logger.warning("")
        logger.warning("⚠️ Некоторые модели не были загружены.")
        logger.warning("Проверьте ошибки выше и попробуйте снова.")
        sys.exit(1)


if __name__ == "__main__":
    main()
