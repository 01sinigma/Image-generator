"""
Скрипт проверки готовности системы к запуску
"""

import sys
import os
from pathlib import Path

print("=" * 60)
print("ПРОВЕРКА ГОТОВНОСТИ СИСТЕМЫ")
print("=" * 60)

errors = []
warnings = []
success = []

# 1. Проверка Python версии
print("\n[1] Проверка Python...")
python_version = sys.version_info
if python_version >= (3, 8):
    success.append(f"Python {python_version.major}.{python_version.minor}.{python_version.micro} - OK")
    print(f"  [OK] Python {python_version.major}.{python_version.minor}.{python_version.micro}")
else:
    errors.append(f"Python {python_version.major}.{python_version.minor} - требуется Python 3.8+")
    print(f"  [ERROR] Python {python_version.major}.{python_version.minor} - требуется 3.8+")

# 2. Проверка необходимых файлов
print("\n[2] Проверка файлов проекта...")
required_files = [
    "generator.py",
    "models.py",
    "app.py",
    "cli.py",
    "config.yaml",
    "requirements.txt"
]

for file in required_files:
    if os.path.exists(file):
        success.append(f"Файл {file} найден")
        print(f"  [OK] {file}")
    else:
        errors.append(f"Файл {file} не найден")
        print(f"  [ERROR] {file} - НЕ НАЙДЕН")

# 3. Проверка зависимостей
print("\n[3] Проверка зависимостей Python...")
dependencies = {
    "torch": "PyTorch",
    "diffusers": "Diffusers",
    "gradio": "Gradio",
    "yaml": "PyYAML",
    "PIL": "Pillow",
    "numpy": "NumPy"
}

for module, name in dependencies.items():
    try:
        if module == "yaml":
            import yaml
        elif module == "PIL":
            from PIL import Image
        else:
            __import__(module)
        success.append(f"{name} установлен")
        print(f"  [OK] {name}")
    except ImportError:
        errors.append(f"{name} не установлен")
        print(f"  [ERROR] {name} - НЕ УСТАНОВЛЕН")

# 4. Проверка CUDA
print("\n[4] Проверка CUDA...")
try:
    import torch
    if torch.cuda.is_available():
        cuda_version = torch.version.cuda
        device_count = torch.cuda.device_count()
        device_name = torch.cuda.get_device_name(0) if device_count > 0 else "Unknown"
        success.append(f"CUDA доступна: {cuda_version}, устройство: {device_name}")
        print(f"  [OK] CUDA доступна: {cuda_version}")
        print(f"  [OK] Устройство: {device_name}")
        print(f"  [OK] Количество GPU: {device_count}")
    else:
        warnings.append("CUDA недоступна - будет использоваться CPU (очень медленно)")
        print(f"  [WARN] CUDA недоступна - будет использоваться CPU")
except ImportError:
    warnings.append("PyTorch не установлен - невозможно проверить CUDA")
    print(f"  [WARN] PyTorch не установлен")

# 5. Проверка моделей
print("\n[5] Проверка моделей...")
models_dir = Path("models")
if models_dir.exists():
    z_image_dir = models_dir / "z-image-turbo"
    qwen_dir = models_dir / "qwen-image-edit"
    
    if z_image_dir.exists():
        model_index = z_image_dir / "model_index.json"
        if model_index.exists():
            success.append("Z-Image-Turbo найдена (частично или полностью)")
            print(f"  [OK] Z-Image-Turbo найдена")
        else:
            warnings.append("Z-Image-Turbo: структура найдена, но модель неполная")
            print(f"  [WARN] Z-Image-Turbo: структура найдена, но модель неполная")
    else:
        warnings.append("Z-Image-Turbo не найдена - будет загружена при первом использовании")
        print(f"  [WARN] Z-Image-Turbo не найдена - будет загружена из интернета")
    
    if qwen_dir.exists():
        model_index = qwen_dir / "model_index.json"
        if model_index.exists():
            success.append("Qwen-Image-Edit найдена (частично или полностью)")
            print(f"  [OK] Qwen-Image-Edit найдена")
        else:
            warnings.append("Qwen-Image-Edit: структура найдена, но модель неполная")
            print(f"  [WARN] Qwen-Image-Edit: структура найдена, но модель неполная")
    else:
        warnings.append("Qwen-Image-Edit не найдена - будет загружена при первом использовании")
        print(f"  [WARN] Qwen-Image-Edit не найдена - будет загружена из интернета")
else:
    warnings.append("Директория models не найдена - модели будут загружены из интернета")
    print(f"  [WARN] Директория models не найдена")

# 6. Проверка конфигурации
print("\n[6] Проверка конфигурации...")
if os.path.exists("config.yaml"):
    try:
        import yaml
        with open("config.yaml", 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        if "models" in config and "z-image-turbo" in config["models"]:
            success.append("Конфигурация валидна")
            print(f"  [OK] config.yaml валиден")
            
            # Проверка настроек для 8GB VRAM
            if config.get("device", {}).get("enable_cpu_offload"):
                success.append("CPU offload включен (для 8GB VRAM)")
                print(f"  [OK] CPU offload включен")
            else:
                warnings.append("CPU offload отключен - может быть проблема с 8GB VRAM")
                print(f"  [WARN] CPU offload отключен")
        else:
            errors.append("Конфигурация неполная")
            print(f"  [ERROR] Конфигурация неполная")
    except Exception as e:
        errors.append(f"Ошибка чтения config.yaml: {e}")
        print(f"  [ERROR] Ошибка чтения config.yaml: {e}")
else:
    errors.append("config.yaml не найден")
    print(f"  [ERROR] config.yaml не найден")

# 7. Проверка директории outputs
print("\n[7] Проверка директорий...")
outputs_dir = Path("outputs")
if not outputs_dir.exists():
    outputs_dir.mkdir(parents=True, exist_ok=True)
    success.append("Директория outputs создана")
    print(f"  [OK] Директория outputs создана")
else:
    success.append("Директория outputs существует")
    print(f"  [OK] Директория outputs существует")

# Итоги
print("\n" + "=" * 60)
print("ИТОГИ ПРОВЕРКИ")
print("=" * 60)

if errors:
    print(f"\n[ERROR] КРИТИЧЕСКИЕ ОШИБКИ ({len(errors)}):")
    for error in errors:
        print(f"  - {error}")
    print("\n[WARN] НЕОБХОДИМО ИСПРАВИТЬ ПЕРЕД ЗАПУСКОМ!")
else:
    print("\n[OK] Критических ошибок не найдено!")

if warnings:
    print(f"\n[WARN] ПРЕДУПРЕЖДЕНИЯ ({len(warnings)}):")
    for warning in warnings:
        print(f"  - {warning}")

if success:
    print(f"\n[OK] УСПЕШНО ({len(success)}):")
    for item in success[:10]:  # Показываем первые 10
        print(f"  - {item}")
    if len(success) > 10:
        print(f"  ... и еще {len(success) - 10}")

print("\n" + "=" * 60)

# Рекомендации
if errors:
    print("\n[INFO] ЧТО НУЖНО СДЕЛАТЬ:")
    print("1. Установите зависимости:")
    print("   pip install -r requirements.txt")
    print("\n2. Установите diffusers из GitHub:")
    print("   pip install git+https://github.com/huggingface/diffusers")
    print("\n3. После установки запустите проверку снова:")
    print("   python check_ready.py")
else:
    print("\n[OK] СИСТЕМА ГОТОВА К ЗАПУСКУ!")
    print("\nДля запуска используйте:")
    print("  python app.py          # Веб-интерфейс")
    print("  python cli.py \"prompt\"  # CLI интерфейс")

print("=" * 60)

