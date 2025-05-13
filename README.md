# 🌲treetypes_classifier_project
Этот репозиторий посвящен проекту по теме: "Распознавание типа растительности на основе анализа космических снимков".
Работа была выполнена в период с 2024-2025 гг.

Данный проект автоматизирует процесс классификации типов растительности по спутниковым данным с использованием моделей машинного обучения. Поддерживается полный pipeline: от склейки TIFF-тайлов и генерации признаков до обучения и инференса модели и визуализации результатов.

[Ссылка на Google Colab, где проводился экспорт cvs-таблицы с признаками и tiff-изображений](https://colab.research.google.com/drive/1ssqD7QBUGZ6zCvrbvjVbeJ3oap_Rj3iP?usp=sharing)

## Установка зависимостей
```bash
pip install -r requirements.txt
```

## Как запустить?
Выполните скрипты в следующем порядке (склейка tif-изображений, добавление признаков разности, обучение модели RF, классификация и визуализация):
```bash
python scripts/merge_tiff.py
python scripts/add_ratio_range.py
python scripts/train_rf.py
python scripts/predict_tiff_rf.py
python scripts/map_vis.py
```
