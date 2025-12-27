# Структура для примеров изображений пресетов

## Текущая структура

Все пресеты теперь имеют структуру:
```python
{
    "Название пресета": {
        "prompt": "текст промпта",
        "image": None  # Будет путь к изображению в будущем
    }
}
```

## Будущая реализация

### Структура папок для примеров
```
preset_examples/
├── quality/
│   ├── ultra_quality.jpg
│   ├── high_quality.jpg
│   ├── masterpiece.jpg
│   └── standard.jpg
├── style/
│   ├── realism.jpg
│   ├── anime.jpg
│   ├── fantasy.jpg
│   └── ...
├── lighting/
│   ├── dramatic.jpg
│   ├── soft.jpg
│   └── ...
├── composition/
│   ├── close_up.jpg
│   ├── medium_shot.jpg
│   └── ...
├── background/
│   ├── no_background.jpg
│   ├── blurred.jpg
│   └── ...
├── location/
│   ├── studio.jpg
│   ├── home.jpg
│   └── ...
└── details/
    ├── macro.jpg
    ├── textures.jpg
    └── ...
```

### Пример обновления PROMPT_PRESETS с изображениями

```python
PROMPT_PRESETS = {
    "quality": {
        "Ultra Quality": {
            "prompt": "ultra detailed, ultra high quality, masterpiece, best quality, extremely detailed",
            "image": "preset_examples/quality/ultra_quality.jpg"
        },
        # ...
    },
    # ...
}
```

### Функция для отображения примеров (будущая реализация)

```python
def create_preset_gallery(category: str):
    """Создание галереи примеров для категории пресетов"""
    examples = []
    for preset_name, preset_data in PROMPT_PRESETS[category].items():
        if preset_data.get("image"):
            examples.append((preset_data["image"], preset_name))
    
    return gr.Gallery(
        value=examples,
        label=f"Примеры: {category}",
        show_label=True,
        elem_id=f"gallery_{category}",
        columns=3,
        rows=2,
        height="auto"
    )
```

### Интеграция в интерфейс (будущая реализация)

```python
# После каждого селектора можно добавить галерею примеров
with gr.Accordion("Примеры качества", open=False):
    quality_gallery = create_preset_gallery("quality")
```

## Текущий статус

✅ Структура данных подготовлена (dict с "prompt" и "image")
✅ Функция `get_preset_prompt()` поддерживает новую структуру
✅ UI элементы добавлены для background и location
✅ Обработчики событий обновлены

## Следующие шаги

1. Создать папку `preset_examples/` с подпапками для каждой категории
2. Добавить примеры изображений для каждого пресета
3. Обновить PROMPT_PRESETS с путями к изображениям
4. Реализовать функцию `create_preset_gallery()`
5. Добавить галереи примеров в интерфейс

