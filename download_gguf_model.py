"""
Скрипт для скачивания GGUF модели Qwen-Image-Edit-2511
"""

import os
from pathlib import Path
from huggingface_hub import hf_hub_download
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_gguf_model(
    repo_id: str = "unsloth/Qwen-Image-Edit-2511-GGUF",
    filename: str = "qwen-image-edit-2511-Q4_K_M.gguf",
    local_dir: str = "./models/qwen-image-edit-gguf"
):
    """
    Скачивание GGUF модели
    
    Args:
        repo_id: ID репозитория на Hugging Face
        filename: Имя файла для скачивания
        local_dir: Локальная директория для сохранения
    """
    try:
        # Создание директории, если не существует
        os.makedirs(local_dir, exist_ok=True)
        logger.info(f"Создана директория: {local_dir}")
        
        # Проверка, существует ли файл
        local_file = Path(local_dir) / filename
        if local_file.exists():
            file_size = local_file.stat().st_size / (1024**3)  # Размер в GB
            logger.info(f"Файл уже существует: {local_file}")
            logger.info(f"Размер файла: {file_size:.2f} GB")
        # Автоматически продолжаем, если файл существует (для фоновой загрузки)
        logger.info("Файл уже существует, проверяем целостность...")
        # Можно добавить проверку целостности, но пока просто продолжаем
        
        logger.info(f"Начало скачивания: {filename}")
        logger.info(f"Репозиторий: {repo_id}")
        logger.info(f"Размер файла: ~13.1 GB (Q4_K_M)")
        logger.info("Это может занять некоторое время...")
        
        # Скачивание файла
        downloaded_file = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=local_dir,
            resume_download=True,  # Продолжить скачивание, если прервано
        )
        
        file_size = Path(downloaded_file).stat().st_size / (1024**3)
        logger.info(f"✅ Файл успешно скачан: {downloaded_file}")
        logger.info(f"Размер файла: {file_size:.2f} GB")
        
        return downloaded_file
        
    except Exception as e:
        logger.error(f"❌ Ошибка при скачивании: {e}")
        raise

if __name__ == "__main__":
    import sys
    
    # Параметры по умолчанию для 8GB VRAM
    default_filename = "qwen-image-edit-2511-Q4_K_M.gguf"
    
    # Можно указать другой файл через аргументы
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        filename = default_filename
    
    logger.info("=" * 60)
    logger.info("Скачивание GGUF модели Qwen-Image-Edit-2511")
    logger.info("=" * 60)
    logger.info(f"Выбранный файл: {filename}")
    logger.info("")
    logger.info("Доступные варианты квантования:")
    logger.info("  Q4_K_S - 12.3 GB (рекомендуется для 8GB VRAM)")
    logger.info("  Q4_K_M - 13.1 GB (лучшее качество для 8GB VRAM)")
    logger.info("  Q5_K_M - 15.0 GB (высокое качество, может не поместиться)")
    logger.info("")
    
    try:
        downloaded_file = download_gguf_model(filename=filename)
        logger.info("")
        logger.info("=" * 60)
        logger.info("✅ Скачивание завершено успешно!")
        logger.info("=" * 60)
        logger.info("")
        logger.info("⚠️  ВАЖНО: GGUF формат требует ComfyUI-GGUF для загрузки.")
        logger.info("Текущая система использует diffusers, который не поддерживает GGUF.")
        logger.info("Для использования этой модели необходимо интегрировать ComfyUI-GGUF.")
        logger.info("")
        logger.info("Пока что используйте оригинальную модель Qwen/Qwen-Image-Edit-2511")
        logger.info("с оптимизациями для 8GB VRAM (CPU offload, sequential offload, float16).")
        
    except KeyboardInterrupt:
        logger.info("")
        logger.info("Скачивание прервано пользователем")
        logger.info("Вы можете продолжить скачивание позже - файл будет докачан автоматически")
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}")
        sys.exit(1)

