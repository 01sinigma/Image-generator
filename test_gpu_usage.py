"""
Тест использования GPU при генерации изображения
"""

import torch
from generator import ZImageGenerator
import time

print("=" * 60)
print("ТЕСТ ИСПОЛЬЗОВАНИЯ GPU")
print("=" * 60)

# Проверка CUDA
print(f"\nCUDA доступна: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Устройство: {torch.cuda.get_device_name(0)}")
    print(f"VRAM всего: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # Очистка кэша перед тестом
    torch.cuda.empty_cache()
    print(f"VRAM свободно (до): {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")

print("\nЗагрузка модели...")
generator = ZImageGenerator()

if torch.cuda.is_available():
    print(f"\nVRAM после загрузки модели:")
    print(f"  Выделено: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    print(f"  Зарезервировано: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
    print(f"  Максимум выделено: {torch.cuda.max_memory_allocated(0) / 1024**3:.2f} GB")

print("\nГенерация тестового изображения...")
start_time = time.time()

try:
    image = generator.generate(
        prompt="A simple test image, red circle on white background",
        height=512,
        width=512,
        num_inference_steps=5,  # Меньше шагов для быстрого теста
        seed=42
    )
    
    elapsed = time.time() - start_time
    
    if torch.cuda.is_available():
        print(f"\nVRAM во время генерации:")
        print(f"  Выделено: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
        print(f"  Зарезервировано: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
        print(f"  Максимум выделено: {torch.cuda.max_memory_allocated(0) / 1024**3:.2f} GB")
    
    print(f"\nВремя генерации: {elapsed:.2f} секунд")
    print(f"Изображение создано: {image.size}")
    print("\n[OK] Генерация успешна!")
    
except Exception as e:
    print(f"\n[ERROR] Ошибка: {e}")
    import traceback
    traceback.print_exc()

print("=" * 60)

