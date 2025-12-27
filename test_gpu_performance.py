"""
Тест производительности GPU при генерации
"""

import torch
import time
from generator import ZImageGenerator

print("=" * 60)
print("ТЕСТ ПРОИЗВОДИТЕЛЬНОСТИ GPU")
print("=" * 60)

# Проверка CUDA
if not torch.cuda.is_available():
    print("CUDA недоступна!")
    exit(1)

print(f"\nУстройство: {torch.cuda.get_device_name(0)}")
print(f"VRAM всего: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# Очистка памяти
torch.cuda.empty_cache()
print(f"VRAM свободно (до): {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / 1024**3:.2f} GB")

print("\nЗагрузка модели...")
generator = ZImageGenerator()

if torch.cuda.is_available():
    allocated = torch.cuda.memory_allocated(0) / 1024**3
    reserved = torch.cuda.memory_reserved(0) / 1024**3
    print(f"\nVRAM после загрузки:")
    print(f"  Выделено: {allocated:.2f} GB")
    print(f"  Зарезервировано: {reserved:.2f} GB")

print("\nГенерация тестового изображения 384x384...")
print("Следите за nvidia-smi в отдельном окне!")

start_time = time.time()

try:
    image = generator.generate(
        prompt="A simple red circle on white background",
        height=384,
        width=384,
        num_inference_steps=9,
        seed=42
    )
    
    elapsed = time.time() - start_time
    
    if torch.cuda.is_available():
        max_allocated = torch.cuda.max_memory_allocated(0) / 1024**3
        print(f"\nVRAM во время генерации:")
        print(f"  Максимум выделено: {max_allocated:.2f} GB")
    
    print(f"\nВремя генерации: {elapsed:.2f} секунд")
    print(f"Скорость: {elapsed/9:.2f} секунд на шаг")
    print(f"Изображение создано: {image.size}")
    
    if elapsed > 120:
        print("\n[WARN] Генерация очень медленная! Возможно проблема с CPU offload.")
    elif elapsed < 30:
        print("\n[OK] Генерация работает нормально!")
    else:
        print("\n[OK] Генерация работает, но можно оптимизировать.")
    
except Exception as e:
    print(f"\n[ERROR] Ошибка: {e}")
    import traceback
    traceback.print_exc()

print("=" * 60)

