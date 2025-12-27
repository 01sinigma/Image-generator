"""
Модуль для детальной проверки полноты модели и докачки недостающих файлов
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from huggingface_hub import hf_hub_download, list_repo_files, repo_info
import time

logger = logging.getLogger(__name__)


class ModelCompletenessChecker:
    """Проверка полноты модели и докачка недостающих файлов"""
    
    def __init__(self, model_name: str, local_path: Optional[str] = None):
        """
        Args:
            model_name: Имя модели на Hugging Face (например, "Qwen/Qwen-Image-Edit-2511")
            local_path: Локальный путь к модели (если есть)
        """
        self.model_name = model_name
        self.local_path = local_path
        self.repo_id = model_name
        self.missing_files = []
        self.incomplete_files = []
        
    def check_model_completeness(self) -> Tuple[bool, Dict]:
        """
        Детальная проверка полноты модели
        
        Returns:
            (is_complete, report) - полная ли модель и детальный отчет
        """
        logger.info("=" * 60)
        logger.info("🔍 Начинаю детальную проверку полноты модели...")
        logger.info(f"Модель: {self.model_name}")
        logger.info("=" * 60)
        
        report = {
            "model_index": False,
            "components": {},
            "total_files": 0,
            "found_files": 0,
            "missing_files": [],
            "incomplete_files": []
        }
        
        # Проверка model_index.json
        model_index_path = None
        if self.local_path:
            model_index_path = os.path.join(self.local_path, "model_index.json")
            if os.path.exists(model_index_path):
                report["model_index"] = True
                logger.info("✅ model_index.json найден")
            else:
                logger.warning("❌ model_index.json отсутствует")
                self.missing_files.append("model_index.json")
        else:
            # Проверяем в кэше Hugging Face
            try:
                from huggingface_hub import scan_cache_dir
                cache_info = scan_cache_dir()
                for repo in cache_info.repos:
                    if self.model_name.split('/')[-1].lower() in str(repo.repo_id).lower():
                        # Модель есть в кэше, но нужно проверить файлы
                        logger.info(f"Модель найдена в кэше: {repo.repo_id}")
                        break
            except:
                pass
        
        # Если model_index.json найден, читаем его для проверки компонентов
        if model_index_path and os.path.exists(model_index_path):
            try:
                with open(model_index_path, 'r', encoding='utf-8') as f:
                    model_index = json.load(f)
                
                # Проверяем каждый компонент
                components_to_check = [
                    "text_encoder", "text_encoder_2",
                    "vae", "vae_decoder", "vae_encoder",
                    "unet", "transformer",
                    "scheduler", "tokenizer", "tokenizer_2"
                ]
                
                for component in components_to_check:
                    if component in model_index:
                        component_path = model_index[component]
                        if isinstance(component_path, list):
                            component_path = component_path[0] if component_path else None
                        
                        if component_path:
                            report["components"][component] = self._check_component(
                                component, component_path
                            )
                            report["total_files"] += report["components"][component].get("total_files", 0)
                            report["found_files"] += report["components"][component].get("found_files", 0)
                            
                            if report["components"][component].get("missing_files"):
                                report["missing_files"].extend(
                                    report["components"][component]["missing_files"]
                                )
                            if report["components"][component].get("incomplete_files"):
                                report["incomplete_files"].extend(
                                    report["components"][component]["incomplete_files"]
                                )
            except Exception as e:
                logger.error(f"Ошибка при чтении model_index.json: {e}")
        
        # Если локального пути нет, проверяем через Hugging Face API
        if not self.local_path:
            logger.info("Локальный путь не указан, проверяю через Hugging Face API...")
            try:
                files_info = list_repo_files(repo_id=self.repo_id, repo_type="model")
                logger.info(f"Найдено {len(files_info)} файлов в репозитории")
                
                # Проверяем наличие критических файлов
                critical_files = [
                    "model_index.json",
                    "scheduler/scheduler_config.json",
                    "text_encoder/config.json"
                ]
                
                for critical_file in critical_files:
                    if any(critical_file in f for f in files_info):
                        logger.info(f"✅ {critical_file} найден в репозитории")
                    else:
                        logger.warning(f"❌ {critical_file} отсутствует в репозитории")
                        self.missing_files.append(critical_file)
            except Exception as e:
                logger.warning(f"Не удалось проверить через API: {e}")
        
        # Итоговый отчет
        is_complete = (
            report["model_index"] and
            len(report["missing_files"]) == 0 and
            len(report["incomplete_files"]) == 0
        )
        
        logger.info("=" * 60)
        logger.info("📊 Итоги проверки:")
        logger.info(f"  Файлов всего: {report['total_files']}")
        logger.info(f"  Файлов найдено: {report['found_files']}")
        logger.info(f"  Файлов отсутствует: {len(report['missing_files'])}")
        logger.info(f"  Файлов неполных: {len(report['incomplete_files'])}")
        logger.info(f"  Модель полная: {'✅ ДА' if is_complete else '❌ НЕТ'}")
        logger.info("=" * 60)
        
        if report["missing_files"]:
            logger.warning("Отсутствующие файлы:")
            for f in report["missing_files"][:10]:  # Показываем первые 10
                logger.warning(f"  - {f}")
            if len(report["missing_files"]) > 10:
                logger.warning(f"  ... и еще {len(report['missing_files']) - 10} файлов")
        
        if report["incomplete_files"]:
            logger.warning("Неполные файлы:")
            for f in report["incomplete_files"][:10]:
                logger.warning(f"  - {f}")
            if len(report["incomplete_files"]) > 10:
                logger.warning(f"  ... и еще {len(report['incomplete_files']) - 10} файлов")
        
        return is_complete, report
    
    def _check_component(self, component_name: str, component_path: str) -> Dict:
        """Проверка конкретного компонента модели"""
        logger.info(f"\n🔍 Проверка компонента: {component_name}")
        
        component_report = {
            "exists": False,
            "total_files": 0,
            "found_files": 0,
            "missing_files": [],
            "incomplete_files": []
        }
        
        if not self.local_path:
            return component_report
        
        # Путь к компоненту
        component_dir = os.path.join(self.local_path, component_name)
        
        if not os.path.exists(component_dir):
            logger.warning(f"  ❌ Директория {component_name} отсутствует")
            component_report["missing_files"].append(f"{component_name}/")
            return component_report
        
        component_report["exists"] = True
        
        # Проверяем index файлы (для sharded моделей)
        index_files = [
            "model.safetensors.index.json",
            "model.safetensors.index",
            "pytorch_model.bin.index.json",
            "pytorch_model.bin.index"
        ]
        
        index_file_path = None
        for index_file in index_files:
            potential_path = os.path.join(component_dir, index_file)
            if os.path.exists(potential_path):
                index_file_path = potential_path
                logger.info(f"  ✅ Найден index файл: {index_file}")
                break
        
        if index_file_path:
            # Это sharded модель - проверяем все файлы из index
            try:
                with open(index_file_path, 'r', encoding='utf-8') as f:
                    index_data = json.load(f)
                
                weight_map = index_data.get("weight_map", {})
                component_report["total_files"] = len(weight_map)
                
                # Получаем уникальные имена файлов
                unique_files = set(weight_map.values())
                component_report["total_files"] = len(unique_files)
                
                logger.info(f"  📦 Sharded модель: {len(unique_files)} файлов весов")
                
                for weight_file in unique_files:
                    weight_file_path = os.path.join(component_dir, weight_file)
                    if os.path.exists(weight_file_path):
                        file_size = os.path.getsize(weight_file_path)
                        if file_size > 0:
                            component_report["found_files"] += 1
                            logger.debug(f"    ✅ {weight_file} ({file_size / (1024**2):.1f} MB)")
                        else:
                            component_report["incomplete_files"].append(f"{component_name}/{weight_file}")
                            logger.warning(f"    ⚠️ {weight_file} пустой (0 bytes)")
                    else:
                        component_report["missing_files"].append(f"{component_name}/{weight_file}")
                        logger.warning(f"    ❌ {weight_file} отсутствует")
            except Exception as e:
                logger.error(f"  ❌ Ошибка при проверке index файла: {e}")
        else:
            # Одиночный файл модели
            single_files = [
                "model.safetensors",
                "pytorch_model.bin",
                "diffusion_pytorch_model.safetensors",
                "diffusion_pytorch_model.bin"
            ]
            
            found_single = False
            for single_file in single_files:
                single_file_path = os.path.join(component_dir, single_file)
                if os.path.exists(single_file_path):
                    file_size = os.path.getsize(single_file_path)
                    component_report["total_files"] = 1
                    if file_size > 0:
                        component_report["found_files"] = 1
                        found_single = True
                        logger.info(f"  ✅ {single_file} найден ({file_size / (1024**2):.1f} MB)")
                        break
                    else:
                        component_report["incomplete_files"].append(f"{component_name}/{single_file}")
                        logger.warning(f"  ⚠️ {single_file} пустой")
                        break
            
            if not found_single:
                component_report["missing_files"].append(f"{component_name}/model.*")
                logger.warning(f"  ❌ Файлы модели не найдены")
        
        # Проверяем config.json
        config_path = os.path.join(component_dir, "config.json")
        if not os.path.exists(config_path):
            component_report["missing_files"].append(f"{component_name}/config.json")
            logger.warning(f"  ❌ config.json отсутствует")
        else:
            logger.info(f"  ✅ config.json найден")
        
        return component_report
    
    def download_missing_files(self, report: Dict, max_retries: int = 3) -> bool:
        """
        Докачка недостающих файлов
        
        Args:
            report: Отчет о проверке модели
            max_retries: Максимальное количество попыток для каждого файла
        
        Returns:
            True если все файлы успешно скачаны
        """
        if not report["missing_files"] and not report["incomplete_files"]:
            logger.info("✅ Все файлы на месте, докачка не требуется")
            return True
        
        logger.info("=" * 60)
        logger.info("📥 Начинаю докачку недостающих файлов...")
        logger.info(f"Файлов для докачки: {len(report['missing_files']) + len(report['incomplete_files'])}")
        logger.info("=" * 60)
        
        all_success = True
        
        # Докачиваем отсутствующие файлы
        for file_path in report["missing_files"]:
            logger.info(f"\n📥 Загрузка: {file_path}")
            success = self._download_file_with_retry(file_path, max_retries)
            if not success:
                all_success = False
                logger.error(f"❌ Не удалось загрузить: {file_path}")
        
        # Перезагружаем неполные файлы
        for file_path in report["incomplete_files"]:
            logger.info(f"\n🔄 Перезагрузка неполного файла: {file_path}")
            # Удаляем неполный файл
            if self.local_path:
                full_path = os.path.join(self.local_path, file_path)
                if os.path.exists(full_path):
                    try:
                        os.remove(full_path)
                        logger.info(f"  Удален неполный файл: {file_path}")
                    except Exception as e:
                        logger.warning(f"  Не удалось удалить файл: {e}")
            
            success = self._download_file_with_retry(file_path, max_retries)
            if not success:
                all_success = False
                logger.error(f"❌ Не удалось перезагрузить: {file_path}")
        
        if all_success:
            logger.info("=" * 60)
            logger.info("✅ Все файлы успешно загружены!")
            logger.info("=" * 60)
        else:
            logger.warning("=" * 60)
            logger.warning("⚠️ Некоторые файлы не удалось загрузить")
            logger.warning("=" * 60)
        
        return all_success
    
    def _download_file_with_retry(self, file_path: str, max_retries: int = 3) -> bool:
        """Загрузка файла с повторными попытками"""
        for attempt in range(1, max_retries + 1):
            try:
                logger.info(f"  Попытка {attempt}/{max_retries}...")
                
                # Определяем локальный путь
                local_file = None
                if self.local_path:
                    local_file = os.path.join(self.local_path, file_path)
                    os.makedirs(os.path.dirname(local_file), exist_ok=True)
                
                # Загружаем файл
                downloaded_path = hf_hub_download(
                    repo_id=self.repo_id,
                    filename=file_path,
                    local_dir=self.local_path,
                    resume_download=True,
                    local_dir_use_symlinks=False
                )
                
                # Проверяем размер файла
                if os.path.exists(downloaded_path):
                    file_size = os.path.getsize(downloaded_path)
                    if file_size > 0:
                        logger.info(f"  ✅ Успешно загружено: {file_path} ({file_size / (1024**2):.1f} MB)")
                        return True
                    else:
                        logger.warning(f"  ⚠️ Файл пустой, повторная попытка...")
                        if os.path.exists(downloaded_path):
                            os.remove(downloaded_path)
                
            except Exception as e:
                logger.warning(f"  ⚠️ Ошибка при загрузке (попытка {attempt}/{max_retries}): {e}")
                if attempt < max_retries:
                    wait_time = 2 ** attempt  # Экспоненциальная задержка: 2s, 4s, 8s
                    logger.info(f"  ⏳ Ожидание {wait_time} секунд перед повторной попыткой...")
                    time.sleep(wait_time)
        
        return False

