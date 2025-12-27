"""
Скрипт для тестирования работы Qwen-Image-Edit-2511 на 8GB VRAM
Проверяет использование памяти, эффективность работы и правильность оптимизаций
"""

import os
import sys
import time
import torch
import logging
from pathlib import Path
from PIL import Image
import yaml

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Импорт генератора
try:
    from models import ModelFactory
except ImportError:
    logger.error("Не удалось импортировать ModelFactory. Убедитесь, что models.py доступен.")
    sys.exit(1)


def get_gpu_memory_info():
    """Получение информации об использовании GPU памяти"""
    if not torch.cuda.is_available():
        return None
    
    device = torch.cuda.current_device()
    memory_allocated = torch.cuda.memory_allocated(device) / (1024**3)  # GB
    memory_reserved = torch.cuda.memory_reserved(device) / (1024**3)  # GB
    memory_total = torch.cuda.get_device_properties(device).total_memory / (1024**3)  # GB
    memory_free = memory_total - memory_reserved
    
    return {
        "allocated": memory_allocated,
        "reserved": memory_reserved,
        "total": memory_total,
        "free": memory_free,
        "usage_percent": (memory_reserved / memory_total) * 100
    }


def print_memory_info(label: str, gpu_info: dict = None):
    """Вывод информации о памяти"""
    print(f"\n{'='*60}")
    print(f"{label}")
    print(f"{'='*60}")
    
    if gpu_info:
        print(f"GPU Memory:")
        print(f"  Выделено:     {gpu_info['allocated']:.2f} GB")
        print(f"  Зарезервировано: {gpu_info['reserved']:.2f} GB")
        print(f"  Всего:        {gpu_info['total']:.2f} GB")
        print(f"  Свободно:     {gpu_info['free']:.2f} GB")
        print(f"  Использование: {gpu_info['usage_percent']:.1f}%")
    
    # Информация о RAM
    try:
        import psutil
        ram = psutil.virtual_memory()
        print(f"\nRAM:")
        print(f"  Использовано: {ram.used / (1024**3):.2f} GB")
        print(f"  Всего:        {ram.total / (1024**3):.2f} GB")
        print(f"  Процент:      {ram.percent:.1f}%")
    except ImportError:
        print("\nRAM: psutil не установлен, информация недоступна")


def check_config():
    """Проверка конфигурации для 8GB VRAM"""
    print("\n" + "="*60)
    print("ПРОВЕРКА КОНФИГУРАЦИИ ДЛЯ 8GB VRAM")
    print("="*60)
    
    try:
        with open("config.yaml", 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Не удалось загрузить config.yaml: {e}")
        return False
    
    # Проверка настроек устройства
    device_config = config.get('device', {})
    enable_cpu_offload = device_config.get('enable_cpu_offload', False)
    sequential_offload = device_config.get('sequential_offload', False)
    
    print(f"\nНастройки устройства:")
    print(f"  enable_cpu_offload: {'✅ ВКЛЮЧЕНО' if enable_cpu_offload else '❌ ВЫКЛЮЧЕНО'}")
    print(f"  sequential_offload: {'✅ ВКЛЮЧЕНО' if sequential_offload else '❌ ВЫКЛЮЧЕНО'}")
    
    if not enable_cpu_offload:
        print("\n⚠️  ВНИМАНИЕ: enable_cpu_offload должен быть ВКЛЮЧЕН для 8GB VRAM!")
        print("   Без этого модель не поместится в память.")
        return False
    
    # Проверка настроек модели Qwen
    qwen_config = config.get('models', {}).get('qwen-image-edit', {})
    torch_dtype = qwen_config.get('torch_dtype', 'float16')
    low_cpu_mem_usage = qwen_config.get('low_cpu_mem_usage', False)
    
    print(f"\nНастройки модели Qwen:")
    print(f"  torch_dtype: {'✅ float16' if torch_dtype == 'float16' else f'⚠️  {torch_dtype} (рекомендуется float16)'}")
    print(f"  low_cpu_mem_usage: {'✅ ВКЛЮЧЕНО' if low_cpu_mem_usage else '❌ ВЫКЛЮЧЕНО'}")
    
    if torch_dtype != 'float16':
        print("\n⚠️  ВНИМАНИЕ: torch_dtype должен быть 'float16' для экономии памяти!")
    
    # Проверка оптимизаций
    opt_config = config.get('optimization', {})
    compile_model = opt_config.get('compile_model', False)
    enable_flash_attention = opt_config.get('enable_flash_attention', False)
    
    print(f"\nОптимизации:")
    print(f"  compile_model: {'❌ ВЫКЛЮЧЕНО' if not compile_model else '⚠️  ВКЛЮЧЕНО (может занимать больше памяти)'}")
    print(f"  enable_flash_attention: {'❌ ВЫКЛЮЧЕНО' if not enable_flash_attention else '⚠️  ВКЛЮЧЕНО (может занимать больше памяти)'}")
    
    print("\n✅ Конфигурация проверена")
    return True


def test_model_loading():
    """Тест загрузки модели"""
    print("\n" + "="*60)
    print("ТЕСТ ЗАГРУЗКИ МОДЕЛИ")
    print("="*60)
    
    # Очистка памяти перед загрузкой
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    initial_memory = get_gpu_memory_info()
    print_memory_info("Память ДО загрузки модели", initial_memory)
    
    try:
        logger.info("Загрузка модели Qwen-Image-Edit-2511...")
        start_time = time.time()
        
        generator = ModelFactory.create_generator('qwen-image-edit')
        
        load_time = time.time() - start_time
        logger.info(f"Модель загружена за {load_time:.2f} секунд")
        
        # Память после загрузки
        loaded_memory = get_gpu_memory_info()
        print_memory_info("Память ПОСЛЕ загрузки модели", loaded_memory)
        
        if loaded_memory:
            memory_used = loaded_memory['reserved'] - initial_memory['reserved']
            print(f"\n📊 Использовано памяти при загрузке: {memory_used:.2f} GB")
            
            if loaded_memory['usage_percent'] > 95:
                print("⚠️  ВНИМАНИЕ: Использование памяти близко к максимуму!")
            elif loaded_memory['usage_percent'] > 85:
                print("✅ Использование памяти в допустимых пределах")
            else:
                print("✅ Использование памяти оптимально")
        
        # Проверка, что CPU offload работает
        if hasattr(generator, 'pipe'):
            pipe = generator.pipe
            # Проверяем, что компоненты не все на GPU
            components_on_gpu = 0
            total_components = 0
            
            for name, component in pipe.components.items():
                total_components += 1
                if hasattr(component, 'device'):
                    if str(component.device).startswith('cuda'):
                        components_on_gpu += 1
            
            print(f"\n📦 Компоненты модели:")
            print(f"  Всего компонентов: {total_components}")
            print(f"  На GPU: {components_on_gpu}")
            print(f"  На CPU (offload): {total_components - components_on_gpu}")
            
            if components_on_gpu < total_components:
                print("✅ CPU offload работает правильно - часть компонентов на CPU")
            else:
                print("⚠️  ВНИМАНИЕ: Все компоненты на GPU, CPU offload может не работать!")
        
        return generator
        
    except Exception as e:
        logger.error(f"Ошибка при загрузке модели: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_image_editing(generator, test_image_path: str = None):
    """Тест редактирования изображения"""
    print("\n" + "="*60)
    print("ТЕСТ РЕДАКТИРОВАНИЯ ИЗОБРАЖЕНИЯ")
    print("="*60)
    
    if generator is None:
        print("❌ Генератор не загружен, пропускаю тест")
        return False
    
    # Создаем тестовое изображение, если не указано
    if test_image_path and os.path.exists(test_image_path):
        image = Image.open(test_image_path)
        print(f"Используется изображение: {test_image_path}")
    else:
        # Создаем простое тестовое изображение
        image = Image.new('RGB', (512, 512), color='red')
        print("Создано тестовое изображение 512x512")
    
    prompt = "Add a blue sky in the background"
    
    # Память перед редактированием
    before_memory = get_gpu_memory_info()
    print_memory_info("Память ПЕРЕД редактированием", before_memory)
    
    try:
        logger.info("Начало редактирования изображения...")
        start_time = time.time()
        
        # Мониторинг использования GPU во время редактирования
        if torch.cuda.is_available():
            # Запускаем мониторинг в отдельном потоке
            import threading
            monitoring = True
            
            def monitor_gpu():
                while monitoring:
                    try:
                        import subprocess
                        result = subprocess.run(
                            ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', 
                             '--format=csv,noheader,nounits'],
                            capture_output=True,
                            text=True,
                            timeout=2
                        )
                        if result.returncode == 0:
                            gpu_util, mem_used, mem_total = result.stdout.strip().split(', ')
                            print(f"  GPU: {gpu_util}% | VRAM: {mem_used}/{mem_total} MB")
                    except:
                        pass
                    time.sleep(2)
            
            monitor_thread = threading.Thread(target=monitor_gpu, daemon=True)
            monitor_thread.start()
        
        # Редактирование
        result = generator.edit(
            images=[image],
            prompt=prompt,
            num_inference_steps=10,  # Меньше шагов для быстрого теста
            guidance_scale=1.0,
            true_cfg_scale=4.0
        )
        
        monitoring = False
        edit_time = time.time() - start_time
        
        # Память после редактирования
        after_memory = get_gpu_memory_info()
        print_memory_info("Память ПОСЛЕ редактирования", after_memory)
        
        print(f"\n⏱️  Время редактирования: {edit_time:.2f} секунд")
        
        if after_memory and before_memory:
            peak_memory = after_memory['reserved'] - before_memory['reserved']
            print(f"📊 Пиковое использование памяти: {peak_memory:.2f} GB")
            
            if after_memory['usage_percent'] > 95:
                print("⚠️  ВНИМАНИЕ: Использование памяти близко к максимуму во время редактирования!")
            else:
                print("✅ Использование памяти в допустимых пределах")
        
        # Сохранение результата
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / f"test_qwen_{int(time.time())}.png"
        result.save(output_path)
        print(f"✅ Результат сохранен: {output_path}")
        
        return True
        
    except torch.cuda.OutOfMemoryError as e:
        logger.error(f"❌ ОШИБКА: Недостаточно VRAM: {e}")
        print("\n💡 Рекомендации:")
        print("  1. Убедитесь, что enable_cpu_offload: true в config.yaml")
        print("  2. Убедитесь, что sequential_offload: true")
        print("  3. Закройте другие приложения, использующие GPU")
        print("  4. Попробуйте уменьшить размер изображения")
        return False
    except Exception as e:
        logger.error(f"Ошибка при редактировании: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Основная функция тестирования"""
    print("="*60)
    print("ТЕСТИРОВАНИЕ QWEN-IMAGE-EDIT-2511 НА 8GB VRAM")
    print("="*60)
    
    # Проверка CUDA
    if not torch.cuda.is_available():
        print("\n❌ CUDA недоступна! Тест требует GPU.")
        return
    
    device_name = torch.cuda.get_device_name(0)
    device_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    
    print(f"\n🎮 GPU: {device_name}")
    print(f"💾 VRAM: {device_memory:.2f} GB")
    
    if device_memory < 7.5:
        print("⚠️  ВНИМАНИЕ: VRAM меньше 8GB, возможны проблемы!")
    elif device_memory > 8.5:
        print("ℹ️  VRAM больше 8GB, тест все равно проверит оптимизации")
    
    # 1. Проверка конфигурации
    if not check_config():
        print("\n❌ Конфигурация не соответствует требованиям для 8GB VRAM")
        print("   Исправьте config.yaml и запустите тест снова")
        return
    
    # 2. Тест загрузки модели
    generator = test_model_loading()
    
    if generator is None:
        print("\n❌ Не удалось загрузить модель. Проверьте логи выше.")
        return
    
    # 3. Тест редактирования
    success = test_image_editing(generator)
    
    # Итоги
    print("\n" + "="*60)
    print("ИТОГИ ТЕСТИРОВАНИЯ")
    print("="*60)
    
    if success:
        print("✅ Все тесты пройдены успешно!")
        print("\nМодель Qwen-Image-Edit-2511 правильно настроена для работы на 8GB VRAM")
    else:
        print("❌ Некоторые тесты не прошли")
        print("\nПроверьте логи выше для диагностики проблем")
    
    # Финальная информация о памяти
    final_memory = get_gpu_memory_info()
    print_memory_info("Финальное состояние памяти", final_memory)


if __name__ == "__main__":
    main()

