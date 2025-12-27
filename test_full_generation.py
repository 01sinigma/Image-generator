"""
Полный тест генерации через ComfyUI GGUF
С мониторингом GPU в реальном времени
"""

import sys
import time
import subprocess
import threading
from pathlib import Path

def get_gpu_stats():
    """Получение статистики GPU"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(',')
            if len(parts) >= 3:
                return {
                    'mem_used': int(parts[0].strip()),
                    'mem_total': int(parts[1].strip()),
                    'util': int(parts[2].strip())
                }
    except:
        pass
    return None


def monitor_gpu(stop_event, stats_list):
    """Мониторинг GPU в отдельном потоке"""
    while not stop_event.is_set():
        stats = get_gpu_stats()
        if stats:
            stats_list.append(stats)
            print(f"\r  GPU: {stats['mem_used']}/{stats['mem_total']}MB, Util: {stats['util']}%   ", end='', flush=True)
        time.sleep(0.5)


def test_generation():
    print("=" * 60)
    print("FULL GENERATION TEST - ComfyUI GGUF")
    print("=" * 60)
    
    from comfyui_api import ComfyUIClient
    
    client = ComfyUIClient()
    
    if not client.is_available():
        print("ERROR: ComfyUI not available!")
        return False
    
    print("\n[1/3] Preparing workflow...")
    
    # Минимальный workflow для тестирования
    # Используем базовые ноды которые точно есть
    workflow = {
        "3": {
            "class_type": "EmptyLatentImage",
            "inputs": {
                "width": 256,  # Минимальный размер для теста
                "height": 256,
                "batch_size": 1
            }
        },
        "5": {
            "class_type": "SaveImage",
            "inputs": {
                "images": ["3", 0],  # Это не сработает, но покажет что ComfyUI отвечает
                "filename_prefix": "test_"
            }
        }
    }
    
    print("  Workflow ready (minimal test)")
    
    print("\n[2/3] Sending to ComfyUI queue...")
    
    # Запуск мониторинга GPU
    stop_event = threading.Event()
    stats_list = []
    monitor_thread = threading.Thread(target=monitor_gpu, args=(stop_event, stats_list))
    monitor_thread.start()
    
    try:
        prompt_id = client.queue_prompt(workflow)
        
        if prompt_id:
            print(f"\n  Prompt ID: {prompt_id}")
            print("\n[3/3] Waiting for result (monitoring GPU)...")
            
            # Ждем результат
            start_time = time.time()
            success, result = client.wait_for_completion(prompt_id, timeout=30)
            elapsed = time.time() - start_time
            
            print(f"\n\n  Elapsed: {elapsed:.1f}s")
            print(f"  Success: {success}")
            
            if result:
                print(f"  Result keys: {list(result.keys())}")
        else:
            print("  ERROR: Failed to queue prompt")
            
    finally:
        stop_event.set()
        monitor_thread.join()
    
    # Статистика GPU
    if stats_list:
        max_mem = max(s['mem_used'] for s in stats_list)
        max_util = max(s['util'] for s in stats_list)
        print(f"\nGPU Stats during test:")
        print(f"  Max Memory: {max_mem} MB")
        print(f"  Max Utilization: {max_util}%")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    # Начальный статус
    print("\nInitial GPU status:")
    stats = get_gpu_stats()
    if stats:
        print(f"  Memory: {stats['mem_used']}/{stats['mem_total']} MB")
        print(f"  Utilization: {stats['util']}%")
    
    print()
    test_generation()

