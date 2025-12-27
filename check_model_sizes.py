"""
Скрипт для проверки размеров моделей на Hugging Face
"""

from huggingface_hub import model_info

def get_model_size(model_id):
    """Получение размера модели"""
    try:
        print(f"\nПроверка модели: {model_id}")
        print("=" * 60)
        
        info = model_info(model_id)
        files = [(f.rfilename, f.size) for f in info.siblings if f.size]
        
        total_size = sum(size for _, size in files)
        total_gb = total_size / (1024 ** 3)
        
        print(f"Общий размер: {total_gb:.2f} GB ({total_size / (1024 ** 2):.2f} MB)")
        print(f"Количество файлов: {len(files)}")
        print("\nКрупнейшие файлы:")
        
        for name, size in sorted(files, key=lambda x: x[1], reverse=True)[:10]:
            size_gb = size / (1024 ** 3)
            size_mb = size / (1024 ** 2)
            print(f"  {name}: {size_gb:.2f} GB ({size_mb:.2f} MB)")
        
        return total_gb
    
    except Exception as e:
        print(f"Ошибка: {e}")
        return None

if __name__ == "__main__":
    print("=" * 60)
    print("Проверка размеров моделей")
    print("=" * 60)
    
    models = {
        "Z-Image-Turbo": "Tongyi-MAI/Z-Image-Turbo",
        "Qwen-Image-Edit-2511": "Qwen/Qwen-Image-Edit-2511"
    }
    
    sizes = {}
    for name, model_id in models.items():
        size = get_model_size(model_id)
        if size:
            sizes[name] = size
    
    print("\n" + "=" * 60)
    print("ИТОГО:")
    print("=" * 60)
    
    total = 0
    for name, size in sizes.items():
        print(f"{name}: {size:.2f} GB")
        total += size
    
    print(f"\nОбщий размер всех моделей: {total:.2f} GB")
    print(f"Рекомендуется иметь минимум: {total * 1.1:.2f} GB свободного места")
    print("=" * 60)

