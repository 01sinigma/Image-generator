@echo off
REM Скрипт запуска приложения с оптимизацией памяти CUDA
REM Устанавливает переменную окружения для лучшего управления памятью

echo Установка переменной окружения для оптимизации памяти CUDA...
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo Запуск приложения...
python app.py

pause

