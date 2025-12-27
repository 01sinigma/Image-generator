"""
Скрипт для диагностики состояния CUDA и GPU
Помогает выявить проблемы с использованием NVIDIA GPU
"""

import torch
import sys
import subprocess
import platform

def check_cuda():
    """Проверка доступности CUDA"""
    print("=" * 60)
    print("ДИАГНОСТИКА CUDA И GPU")
    print("=" * 60)
    
    # 1. Проверка PyTorch
    print("\n[1] Информация о PyTorch:")
    print(f"  Версия PyTorch: {torch.__version__}")
    print(f"  Версия CUDA в PyTorch: {torch.version.cuda if torch.version.cuda else 'N/A'}")
    print(f"  cuDNN версия: {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else 'N/A'}")
    
    # 2. Проверка доступности CUDA
    print("\n[2] Доступность CUDA:")
    cuda_available = torch.cuda.is_available()
    print(f"  CUDA доступна: {cuda_available}")
    
    if not cuda_available:
        print("\n  [ERROR] CUDA недоступна!")
        print("  Возможные причины:")
        print("    - Установлена CPU-версия PyTorch")
        print("    - Драйверы NVIDIA не установлены или устарели")
        print("    - GPU не поддерживает CUDA")
        print("\n  Решение:")
        print("    - Установите PyTorch с поддержкой CUDA:")
        print("      pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
        return False
    
    # 3. Информация об устройствах
    print("\n[3] Информация об устройствах:")
    device_count = torch.cuda.device_count()
    print(f"  Количество GPU: {device_count}")
    
    for i in range(device_count):
        print(f"\n  GPU {i}:")
        try:
            device_name = torch.cuda.get_device_name(i)
            print(f"    Название: {device_name}")
            
            # Проверка возможности использования устройства
            try:
                torch.cuda.set_device(i)
                current_device = torch.cuda.current_device()
                print(f"    Текущее устройство: {current_device}")
                
                # Проверка памяти
                memory_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
                memory_allocated = torch.cuda.memory_allocated(i) / 1024**3
                memory_reserved = torch.cuda.memory_reserved(i) / 1024**3
                memory_free = memory_total - memory_reserved
                
                print(f"    Память:")
                print(f"      Всего: {memory_total:.2f} GB")
                print(f"      Выделено: {memory_allocated:.2f} GB")
                print(f"      Зарезервировано: {memory_reserved:.2f} GB")
                print(f"      Свободно: {memory_free:.2f} GB")
                
                # Проверка вычислительных возможностей
                compute_capability = torch.cuda.get_device_capability(i)
                print(f"    Compute Capability: {compute_capability[0]}.{compute_capability[1]}")
                
            except Exception as e:
                print(f"    [ERROR] Ошибка при работе с устройством: {e}")
                return False
                
        except Exception as e:
            print(f"    [ERROR] Ошибка получения информации об устройстве: {e}")
            return False
    
    # 4. Тест простой операции на GPU
    print("\n[4] Тест простой операции на GPU:")
    try:
        torch.cuda.set_device(0)
        torch.cuda.synchronize()
        
        # Создание простого тензора на GPU
        test_tensor = torch.randn(100, 100).cuda()
        result = torch.matmul(test_tensor, test_tensor)
        torch.cuda.synchronize()
        
                print("  [OK] Тест пройден успешно!")
                print("  GPU может выполнять вычисления")
        
        # Очистка
        del test_tensor, result
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"  [ERROR] Тест не пройден: {e}")
        print("  GPU не может выполнять вычисления")
        return False
    
    # 5. Проверка процессов, использующих GPU
    print("\n[5] Проверка процессов, использующих GPU:")
    if platform.system() == "Windows":
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0 and result.stdout.strip():
                print("  Процессы, использующие GPU:")
                for line in result.stdout.strip().split('\n'):
                    if line.strip():
                        print(f"    {line}")
            else:
                print("  Нет активных процессов на GPU")
        except FileNotFoundError:
            print("  nvidia-smi не найден (возможно, драйверы не установлены)")
        except Exception as e:
            print(f"  Не удалось проверить процессы: {e}")
    else:
        print("  Проверка процессов доступна только на Windows")
    
    # 6. Рекомендации
    print("\n[6] Рекомендации:")
    if cuda_available:
        print("  [OK] CUDA доступна и работает")
        print("  [OK] GPU готов к использованию")
        print("\n  Если возникают ошибки 'CUDA device is busy':")
        print("    - Закройте другие приложения, использующие GPU")
        print("    - Перезапустите Python процесс")
        print("    - Проверьте, не запущены ли другие экземпляры приложения")
    else:
        print("  [ERROR] Требуется установка PyTorch с поддержкой CUDA")
    
    print("\n" + "=" * 60)
    return cuda_available

if __name__ == "__main__":
    try:
        success = check_cuda()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n[ERROR] Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

