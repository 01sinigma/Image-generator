"""
Установщик ComfyUI с поддержкой GGUF для Qwen-Image-Edit-2511

Автоматически:
1. Клонирует ComfyUI
2. Устанавливает ComfyUI-GGUF extension
3. Настраивает пути к моделям
4. Создаёт workflow для Qwen GGUF
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Пути
BASE_DIR = Path(__file__).parent
COMFYUI_DIR = BASE_DIR / "ComfyUI"
GGUF_MODEL_PATH = BASE_DIR / "models" / "qwen-image-edit-gguf" / "qwen-image-edit-2511-Q4_K_M.gguf"


def run_command(cmd, cwd=None, check=True):
    """Выполнение команды"""
    logger.info(f"Выполняю: {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0 and check:
        logger.error(f"Ошибка: {result.stderr}")
        return False
    return True


def check_git():
    """Проверка наличия Git"""
    try:
        subprocess.run(["git", "--version"], capture_output=True, check=True)
        return True
    except:
        logger.error("Git не найден! Установите Git: https://git-scm.com/")
        return False


def install_comfyui():
    """Установка ComfyUI"""
    logger.info("\n" + "="*60)
    logger.info("Шаг 1: Установка ComfyUI")
    logger.info("="*60 + "\n")
    
    if COMFYUI_DIR.exists():
        logger.info(f"ComfyUI уже установлен: {COMFYUI_DIR}")
        return True
    
    if not check_git():
        return False
    
    # Клонирование ComfyUI
    logger.info("Клонирование ComfyUI...")
    if not run_command(f'git clone https://github.com/comfyanonymous/ComfyUI.git "{COMFYUI_DIR}"'):
        return False
    
    # Установка зависимостей
    logger.info("\nУстановка зависимостей ComfyUI...")
    requirements_file = COMFYUI_DIR / "requirements.txt"
    if requirements_file.exists():
        if not run_command(f'pip install -r "{requirements_file}"'):
            logger.warning("Некоторые зависимости не установились, но продолжаем...")
    
    logger.info("✅ ComfyUI установлен!")
    return True


def install_gguf_extension():
    """Установка ComfyUI-GGUF extension"""
    logger.info("\n" + "="*60)
    logger.info("Шаг 2: Установка ComfyUI-GGUF Extension")
    logger.info("="*60 + "\n")
    
    custom_nodes_dir = COMFYUI_DIR / "custom_nodes"
    gguf_dir = custom_nodes_dir / "ComfyUI-GGUF"
    
    if gguf_dir.exists():
        logger.info(f"ComfyUI-GGUF уже установлен: {gguf_dir}")
        return True
    
    custom_nodes_dir.mkdir(parents=True, exist_ok=True)
    
    # Клонирование ComfyUI-GGUF
    logger.info("Клонирование ComfyUI-GGUF...")
    if not run_command(f'git clone https://github.com/city96/ComfyUI-GGUF.git "{gguf_dir}"'):
        return False
    
    # Установка зависимостей GGUF
    gguf_requirements = gguf_dir / "requirements.txt"
    if gguf_requirements.exists():
        logger.info("Установка зависимостей GGUF...")
        run_command(f'pip install -r "{gguf_requirements}"', check=False)
    
    # Установка gguf библиотеки
    logger.info("Установка gguf библиотеки...")
    run_command("pip install gguf", check=False)
    
    logger.info("✅ ComfyUI-GGUF установлен!")
    return True


def setup_model_links():
    """Настройка путей к моделям"""
    logger.info("\n" + "="*60)
    logger.info("Шаг 3: Настройка путей к моделям")
    logger.info("="*60 + "\n")
    
    # Директории для моделей в ComfyUI
    comfy_models = COMFYUI_DIR / "models"
    unet_dir = comfy_models / "unet"
    checkpoints_dir = comfy_models / "checkpoints"
    
    unet_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    
    # Проверка наличия GGUF модели
    if not GGUF_MODEL_PATH.exists():
        logger.warning(f"⚠️ GGUF модель не найдена: {GGUF_MODEL_PATH}")
        logger.info("Скачайте модель через: python download_all_models.py --model qwen")
        return False
    
    # Создание символической ссылки или копии
    target_path = unet_dir / GGUF_MODEL_PATH.name
    
    if target_path.exists():
        logger.info(f"Модель уже настроена: {target_path}")
        return True
    
    try:
        # Пробуем создать символическую ссылку (требует прав администратора на Windows)
        target_path.symlink_to(GGUF_MODEL_PATH)
        logger.info(f"✅ Создана ссылка: {target_path}")
    except OSError:
        # Если не получилось, создаём extra_model_paths.yaml
        logger.info("Создание конфигурации путей к моделям...")
        
        extra_paths = COMFYUI_DIR / "extra_model_paths.yaml"
        config = f"""
# Пути к внешним моделям
image_generator:
    base_path: {BASE_DIR.as_posix()}
    unet: models/qwen-image-edit-gguf
    checkpoints: models/qwen-image-edit-gguf
    loras: models/lora
"""
        with open(extra_paths, 'w', encoding='utf-8') as f:
            f.write(config)
        
        logger.info(f"✅ Создан конфиг: {extra_paths}")
    
    return True


def create_qwen_workflow():
    """Создание workflow для Qwen GGUF"""
    logger.info("\n" + "="*60)
    logger.info("Шаг 4: Создание Workflow для Qwen GGUF")
    logger.info("="*60 + "\n")
    
    workflows_dir = COMFYUI_DIR / "user" / "default" / "workflows"
    workflows_dir.mkdir(parents=True, exist_ok=True)
    
    # Базовый workflow для GGUF
    workflow = {
        "last_node_id": 10,
        "last_link_id": 10,
        "nodes": [
            {
                "id": 1,
                "type": "UnetLoaderGGUF",
                "pos": [100, 100],
                "size": [300, 100],
                "properties": {},
                "widgets_values": ["qwen-image-edit-2511-Q4_K_M.gguf"]
            },
            {
                "id": 2,
                "type": "CLIPLoaderGGUF",
                "pos": [100, 250],
                "size": [300, 100],
                "properties": {},
                "widgets_values": [""]
            },
            {
                "id": 3,
                "type": "EmptyLatentImage",
                "pos": [100, 400],
                "size": [300, 100],
                "properties": {},
                "widgets_values": [512, 512, 1]
            },
            {
                "id": 4,
                "type": "CLIPTextEncode",
                "pos": [450, 100],
                "size": [400, 150],
                "properties": {},
                "widgets_values": ["A beautiful landscape, high quality, detailed"]
            },
            {
                "id": 5,
                "type": "CLIPTextEncode", 
                "pos": [450, 300],
                "size": [400, 150],
                "properties": {},
                "widgets_values": ["blurry, low quality, bad"]
            },
            {
                "id": 6,
                "type": "KSampler",
                "pos": [900, 200],
                "size": [300, 250],
                "properties": {},
                "widgets_values": [
                    0,  # seed
                    "randomize",
                    4,  # steps (Lightning LoRA style)
                    1.0,  # cfg
                    "euler",
                    "normal",
                    1.0  # denoise
                ]
            },
            {
                "id": 7,
                "type": "VAEDecode",
                "pos": [1250, 200],
                "size": [200, 100],
                "properties": {}
            },
            {
                "id": 8,
                "type": "SaveImage",
                "pos": [1500, 200],
                "size": [300, 300],
                "properties": {},
                "widgets_values": ["ComfyUI"]
            }
        ],
        "links": [],
        "groups": [],
        "config": {},
        "extra": {
            "ds": {
                "scale": 1,
                "offset": [0, 0]
            }
        },
        "version": 0.4
    }
    
    workflow_path = workflows_dir / "qwen_gguf_workflow.json"
    with open(workflow_path, 'w', encoding='utf-8') as f:
        json.dump(workflow, f, indent=2)
    
    logger.info(f"✅ Workflow создан: {workflow_path}")
    
    # Также сохраняем в корень проекта для удобства
    local_workflow = BASE_DIR / "comfyui_qwen_workflow.json"
    with open(local_workflow, 'w', encoding='utf-8') as f:
        json.dump(workflow, f, indent=2)
    
    logger.info(f"✅ Копия workflow: {local_workflow}")
    
    return True


def create_start_scripts():
    """Создание скриптов запуска"""
    logger.info("\n" + "="*60)
    logger.info("Шаг 5: Создание скриптов запуска")
    logger.info("="*60 + "\n")
    
    # Скрипт запуска ComfyUI
    start_comfyui = BASE_DIR / "start_comfyui.bat"
    content = f'''@echo off
REM Запуск ComfyUI с поддержкой GGUF
echo.
echo ============================================================
echo    ComfyUI + GGUF для Qwen-Image-Edit-2511
echo ============================================================
echo.

cd /d "{COMFYUI_DIR}"

REM Оптимизация памяти
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo Запуск ComfyUI...
echo Откройте в браузере: http://127.0.0.1:8188
echo.

python main.py --listen --lowvram

pause
'''
    with open(start_comfyui, 'w', encoding='utf-8') as f:
        f.write(content)
    
    logger.info(f"✅ Создан: {start_comfyui}")
    
    # Скрипт запуска с API
    start_api = BASE_DIR / "start_comfyui_api.bat"
    api_content = f'''@echo off
REM Запуск ComfyUI в режиме API
cd /d "{COMFYUI_DIR}"
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python main.py --listen --lowvram --enable-cors-header
pause
'''
    with open(start_api, 'w', encoding='utf-8') as f:
        f.write(api_content)
    
    logger.info(f"✅ Создан: {start_api}")
    
    return True


def main():
    """Главная функция установки"""
    logger.info("""
╔══════════════════════════════════════════════════════════════╗
║      Установка ComfyUI + GGUF для Qwen-Image-Edit-2511       ║
║              Оптимизировано для 8GB VRAM                     ║
╚══════════════════════════════════════════════════════════════╝
""")
    
    # Проверка GGUF модели
    if GGUF_MODEL_PATH.exists():
        size_gb = GGUF_MODEL_PATH.stat().st_size / (1024**3)
        logger.info(f"✅ GGUF модель найдена: {size_gb:.2f} GB")
    else:
        logger.warning(f"⚠️ GGUF модель не найдена!")
        logger.info(f"   Ожидаемый путь: {GGUF_MODEL_PATH}")
        logger.info("   Скачайте через: python download_all_models.py --model qwen")
    
    steps = [
        ("ComfyUI", install_comfyui),
        ("GGUF Extension", install_gguf_extension),
        ("Модели", setup_model_links),
        ("Workflow", create_qwen_workflow),
        ("Скрипты", create_start_scripts),
    ]
    
    success = True
    for name, func in steps:
        try:
            if not func():
                logger.error(f"❌ Ошибка на шаге: {name}")
                success = False
                break
        except Exception as e:
            logger.error(f"❌ Исключение на шаге {name}: {e}")
            success = False
            break
    
    logger.info("\n" + "="*60)
    if success:
        logger.info("✅ УСТАНОВКА ЗАВЕРШЕНА УСПЕШНО!")
        logger.info("="*60)
        logger.info("""
Для запуска ComfyUI:
  1. Запустите: start_comfyui.bat
  2. Откройте: http://127.0.0.1:8188
  3. Загрузите workflow: comfyui_qwen_workflow.json

Для интеграции с основным приложением:
  1. Запустите: start_comfyui_api.bat
  2. Затем: python app.py
  3. Выберите режим "ComfyUI" в интерфейсе
""")
    else:
        logger.info("❌ УСТАНОВКА ЗАВЕРШЕНА С ОШИБКАМИ")
        logger.info("="*60)
        logger.info("Проверьте логи выше и повторите попытку")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())

