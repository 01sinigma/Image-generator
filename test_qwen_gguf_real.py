"""
Реальный тест генерации Qwen GGUF через ComfyUI
С мониторингом GPU
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
            bar_len = int(stats['mem_used'] / stats['mem_total'] * 20)
            bar = '#' * bar_len + '-' * (20 - bar_len)
            print(f"\r  GPU [{bar}] {stats['mem_used']}/{stats['mem_total']}MB, Util: {stats['util']:3d}%   ", end='', flush=True)
        time.sleep(0.5)


def test_real_generation():
    print("=" * 60)
    print("REAL GENERATION TEST - Qwen GGUF via ComfyUI")
    print("=" * 60)
    
    from comfyui_api import ComfyUIClient, ComfyUIGGUFGenerator
    
    # Проверка доступности
    client = ComfyUIClient()
    if not client.is_available():
        print("ERROR: ComfyUI not available!")
        return False
    
    print("\n[1/5] ComfyUI is available")
    
    # Создание генератора
    print("\n[2/5] Creating generator...")
    generator = ComfyUIGGUFGenerator(
        model_name="qwen-image-edit-2511-Q4_K_M.gguf"
    )
    print("  Generator ready")
    
    # Параметры генерации
    prompt = "A beautiful red rose on white background, high quality, detailed"
    print(f"\n[3/5] Test prompt: '{prompt}'")
    print("  Size: 256x256 (minimal for fast test)")
    print("  Steps: 4")
    
    # Запуск мониторинга GPU
    stop_event = threading.Event()
    stats_list = []
    monitor_thread = threading.Thread(target=monitor_gpu, args=(stop_event, stats_list))
    monitor_thread.start()
    
    print("\n[4/5] Generating...")
    start_time = time.time()
    
    try:
        image = generator.generate(
            prompt=prompt,
            height=256,
            width=256,
            num_inference_steps=4,
            guidance_scale=1.0,
            seed=42
        )
        
        elapsed = time.time() - start_time
        
    finally:
        stop_event.set()
        monitor_thread.join()
    
    print(f"\n\n[5/5] Result:")
    print(f"  Elapsed time: {elapsed:.1f}s")
    
    if image:
        print(f"  Image size: {image.size}")
        
        # Сохранение
        output_path = Path("outputs") / "test_qwen_gguf.png"
        output_path.parent.mkdir(exist_ok=True)
        image.save(output_path)
        print(f"  Saved to: {output_path}")
        
        success = True
    else:
        print("  ERROR: No image generated!")
        success = False
    
    # Статистика GPU
    if stats_list:
        max_mem = max(s['mem_used'] for s in stats_list)
        max_util = max(s['util'] for s in stats_list)
        avg_util = sum(s['util'] for s in stats_list) / len(stats_list)
        
        print(f"\nGPU Statistics:")
        print(f"  Peak Memory: {max_mem} MB")
        print(f"  Peak Utilization: {max_util}%")
        print(f"  Avg Utilization: {avg_util:.1f}%")
        
        if max_util > 10:
            print("  -> GPU was actively used during generation!")
        else:
            print("  -> Warning: Low GPU utilization")
    
    print("\n" + "=" * 60)
    if success:
        print("SUCCESS! Qwen GGUF generation works correctly!")
    else:
        print("FAILED: Check ComfyUI logs for errors")
    print("=" * 60)
    
    return success


if __name__ == "__main__":
    # Начальный статус
    print("\nInitial GPU status:")
    stats = get_gpu_stats()
    if stats:
        print(f"  Memory: {stats['mem_used']}/{stats['mem_total']} MB")
        print(f"  Utilization: {stats['util']}%")
    
    print()
    success = test_real_generation()
    
    # Финальный статус
    print("\nFinal GPU status:")
    stats = get_gpu_stats()
    if stats:
        print(f"  Memory: {stats['mem_used']}/{stats['mem_total']} MB")
        print(f"  Utilization: {stats['util']}%")
    
    sys.exit(0 if success else 1)

