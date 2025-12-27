# Исправление ошибки CPU Offload

## Проблема
Ошибка: `enable_model_cpu_offload requires accelerator, but not found`

## Причина
CPU offload работает только с CUDA. Когда CUDA недоступна, нужно просто загружать модель на CPU без offload.

## Исправления

### 1. Обновлен models.py
- Исправлена проверка: CPU offload включается только если CUDA доступна
- Исправлено предупреждение: `torch_dtype` заменен на `dtype`
- Добавлена автоматическая проверка устройства

### 2. Обновлен config.yaml
- `device.type` изменен на `"auto"` для автоматического определения

## Что изменилось

**До:**
```python
if self.config['device'].get('enable_cpu_offload', False):
    self.pipe.enable_model_cpu_offload()  # Ошибка на CPU
```

**После:**
```python
if self.config['device'].get('enable_cpu_offload', False) and self.device == 'cuda':
    try:
        self.pipe.enable_model_cpu_offload()
    except Exception as e:
        logger.warning(f"Не удалось включить CPU offload: {e}")
        self.pipe.to(self.device)
else:
    self.pipe.to(self.device)  # Просто загружаем на CPU
```

## Теперь можно запускать

```bash
python app.py
```

Модель будет работать на CPU (медленно, но работает).

