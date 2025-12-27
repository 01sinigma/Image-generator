# 🚀 Быстрые улучшения для решения проблем загрузки

## ТОП-3 критичных улучшения (внедрить в первую очередь)

### 1. Retry механизм ⭐⭐⭐

**Проблема:** При сетевом сбое загрузка полностью прерывается.

**Быстрое решение:**
```python
# Добавить в models.py
from tenacity import retry, stop_after_attempt, wait_exponential
import requests

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=5, max=60),
    reraise=True
)
def _load_with_retry(self, model_name, **kwargs):
    """Загрузка модели с автоматическими повторами"""
    try:
        return QwenImageEditPlusPipeline.from_pretrained(model_name, **kwargs)
    except (requests.exceptions.Timeout, 
            requests.exceptions.ConnectionError,
            OSError) as e:
        logger.warning(f"Ошибка загрузки: {e}. Повторная попытка...")
        raise  # tenacity перехватит и повторит
```

**Установка:** `pip install tenacity`

---

### 2. Проверка перед загрузкой ⭐⭐⭐

**Проблема:** Загрузка начинается даже при проблемах.

**Быстрое решение:**
```python
# Добавить в models.py перед _load_pipeline()

def _preload_checks(self):
    """Проверки перед началом загрузки"""
    import shutil
    import requests
    
    # Проверка сети
    try:
        response = requests.get("https://huggingface.co", timeout=5)
        if response.status_code != 200:
            raise ConnectionError("Hugging Face недоступен")
    except Exception as e:
        raise ConnectionError(f"Нет интернет-соединения: {e}")
    
    # Проверка места на диске
    required_gb = 50 if 'qwen' in self.model_config.get('name', '').lower() else 10
    stat = shutil.disk_usage(".")
    free_gb = stat.free / (1024**3)
    
    if free_gb < required_gb:
        raise ValueError(
            f"Недостаточно места на диске!\n"
            f"Требуется: {required_gb}GB\n"
            f"Доступно: {free_gb:.2f}GB"
        )
    
    logger.info(f"✅ Проверки пройдены: сеть OK, место на диске OK ({free_gb:.2f}GB)")
```

---

### 3. Мониторинг скорости загрузки ⭐⭐

**Проблема:** Непонятно, идет ли загрузка и сколько осталось.

**Быстрое решение:**
```python
# Улучшить monitor.py - добавить в ProcessMonitor

def _monitor_download_speed(self):
    """Мониторинг скорости загрузки через размер кэша"""
    try:
        from huggingface_hub import scan_cache_dir
        cache_info = scan_cache_dir()
        
        for repo in cache_info.repos:
            if 'qwen' in str(repo.repo_id).lower():
                # Проверяем изменение размера
                current_size = repo.size_on_disk
                if hasattr(self, 'last_cache_size'):
                    delta = current_size - self.last_cache_size
                    if delta > 0:
                        elapsed = time.time() - self.last_cache_check
                        speed_mb_s = (delta / elapsed) / (1024**2)
                        self._log_status(f"📥 Скорость загрузки: {speed_mb_s:.2f} MB/s")
                
                self.last_cache_size = current_size
                self.last_cache_check = time.time()
                break
    except:
        pass
```

---

## 📝 Что добавить в config.yaml

```yaml
# Добавить в конец config.yaml

download:
  # Retry настройки
  max_retries: 5
  retry_delay_min: 5  # секунды
  retry_delay_max: 60  # секунды
  
  # Таймауты
  timeout: 300  # 5 минут для обычных запросов
  large_file_timeout: 1800  # 30 минут для больших файлов (>10GB)
  
  # Проверки перед загрузкой
  check_network: true
  check_disk_space: true
  required_disk_space_gb: 50  # Для Qwen, 10 для других
  
  # Мониторинг
  show_download_speed: true
  show_eta: true
```

---

## 🔧 Порядок внедрения (30 минут)

1. **Установить зависимости** (2 мин):
   ```bash
   pip install tenacity requests
   ```

2. **Добавить проверки** (10 мин):
   - Добавить `_preload_checks()` в `BaseImageGenerator`
   - Вызывать перед загрузкой в `_load_pipeline()`

3. **Добавить retry** (15 мин):
   - Обернуть `from_pretrained()` в retry декоратор
   - Добавить обработку специфичных исключений

4. **Улучшить мониторинг** (3 мин):
   - Добавить отображение скорости в `ProcessMonitor`

---

## ✅ Ожидаемый результат

После внедрения:
- ✅ Автоматические повторы при сбоях сети
- ✅ Предупреждение о проблемах ДО начала загрузки
- ✅ Видимость скорости загрузки
- ✅ Меньше зависаний и неожиданных прерываний

---

## 🎯 Следующие шаги (после быстрых исправлений)

1. Проверка целостности файлов
2. Кэширование прогресса
3. Параллельная загрузка компонентов
4. Детальная диагностика

См. `IMPROVEMENTS_RECOMMENDATIONS.md` для полного списка улучшений.

