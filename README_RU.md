# SnnForMI

[English](README.md) | Russian

SnnForMI - исследовательский проект для классификации моторного воображения по ЭЭГ. Основной акцент сделан на легковесные спайковые нейронные сети, которые могут быть полезны для компактного и энергоэффективного вывода, но в том же экспериментальном окружении оставлены классические EEG-baseline модели для сравнения.

Проект был реорганизован вокруг понятного экспериментального пайплайна: загрузка и препроцессинг данных находятся в `src/data`, архитектуры моделей - в `src/models`, а скрипты обучения и подбора параметров - в `src/training`. Файлы в корне оставлены как совместимые запускатели, поэтому команды вроде `python main.py ...` продолжают работать.

## Референсная Работа

Основная референсная статья для легковесной SNN-модели:

> H. Zhang, H. Wang, J. An, S. Zheng, and D. Wu, "A lightweight spiking neural network for EEG-based motor imagery classification," Key Laboratory of the Ministry of Education for Image Processing and Intelligent Control, School of Artificial Intelligence and Automation, Huazhong University of Science and Technology, Wuhan 430074, China.

Замечания по реализации:

- Референсная реализация, использованная для ориентира: [`Zhr1110/SnnForMI`](https://github.com/Zhr1110/SnnForMI). Этот репозиторий является публичной реализацией к статье про lightweight SNN и содержит исходную структуру экспериментов с `preprocess.py`, `preprocess_for_filterbank.py`, `main.py` и моделью на основе `CUPY_SNN_PLIF`.
- Референсная модель `CUPY_SNN_PLIF` следует PLIF-style дизайну нейрона из статьи про lightweight SNN.
- CuPy-backed PLIF/LIF нейронные модули реализованы в `tools/neurons.py`.
- SNN-архитектуры проекта описаны в `src/models/SNNs.py`.
- SpikingJelly использован как референс реализации для динамики нейронов, surrogate-gradient обучения, step-mode utilities и CUDA-oriented spiking modules.
- Дополнительные baseline-модели: EEGNet, ShallowConvNet, DeepConvNet, FBCNet и Norse latency-coded SNN.

По сравнению с референсным репозиторием эта версия реорганизует код в `src/data`, `src/models` и `src/training`, сохраняет совместимые wrappers в корне, добавляет дополнительные evaluation utilities и фиксирует локальные benchmark-артефакты для экспериментов диссертационной работы.

## Снимок Результатов

Значения ниже взяты из сохраненных артефактов в `benchmarks/`. Их стоит читать как записи локальных экспериментов проекта, а не как универсальный leaderboard.

| Датасет / режим | Модель | Accuracy | Macro-F1 | Источник |
|---|---:|---:|---:|---|
| BNCI2014-001 | LIF readout, без subject tuning | 69.10% | 69.03% | `full_eval_lif_readout_best_seed26.json` |
| BNCI2014-001 | LIF readout, subject tuning | 69.98% | 69.91% | `full_eval_subject_tuning_lif_seed26_selected.json` |
| BNCI2014-001 | LIF readout, best merged selection | 71.06% | 71.01% | `full_eval_subject_tuning_lif_seed26_final_merged.json` |
| BNCI2014-001 | Lightweight SNN reference row | 68.92% | n/a | `bnci2014_001_accuracy_f1_plot_data.csv` |
| Weibo2014 | LIF readout, без tuning | 54.30% | 54.08% | `full_eval_weibo2014_lif_readout_best_seed28.json` |
| Weibo2014 | LIF readout, subject tuning | 54.56% | 54.40% | `full_eval_weibo2014_lif_seed28_combined_tune.json` |
| Weibo2014 | LIF readout, target merged selection | 55.57% | 55.46% | `full_eval_weibo2014_lif_seed28_target_merged.json` |
| Weibo2014 | Lightweight SNN reference row | 56.64% | n/a | `weibo2014_accuracy_f1_plot_data.csv` |

Снимок эффективности для BNCI2014-001:

| Модель | Обучаемые параметры | Energy / sample | Avg. power | Inference time |
|---|---:|---:|---:|---:|
| `CUPY_SNN_PLIF` | 2,601 | 0.825 mJ | 82.299 W | 0.02599 s |
| `CUPY_SNN_LIF_READOUT` | 2,609 | 1.038 mJ | 82.299 W | 0.03268 s |
| `EEGNet` | 3,476 | 2.748 mJ | 161.946 W | 0.04398 s |
| `ShallowConvNet` | 46,084 | 47.466 mJ | 113.738 W | 1.08172 s |
| `DeepConvNet` | 320,479 | 45.571 mJ | 167.663 W | 0.70451 s |

## Структура Проекта

- `src/data/`: загрузчики датасетов, препроцессинг, Euclidean alignment, filterbank-подготовка и PyTorch dataset wrapper.
- `src/models/`: архитектуры EEGNet, ConvNet, FBCNet, Norse и CuPy-based SNN.
- `src/training/`: основные training entry points, grid runner'ы, subject tuning и скрипты обучения для отдельных наборов данных.
- `tools/`: низкоуровневый spiking backend, CUDA helpers, surrogate gradients, аугментации и compatibility wrappers.
- `benchmarks/results/`: отобранные JSON/CSV summaries для таблиц README и plotting scripts.
- `benchmarks/figures/`: сгенерированные графики, сгруппированные по типу benchmark'а.
- `benchmarks/runs/`: локальные сырые логи запусков и scratch experiment traces; обычно не хранятся в Git history.
- `data/`: raw и preprocessed файлы датасетов.

## Подготовка Данных

Подготовка single-band EEG представления:

```bash
python preprocess.py --dataset 0 --resample_fs 250
```

Подготовка filterbank-данных для FBCNet:

```bash
python preprocess_for_filterbank.py --dataset 2 --resample_fs 250
```

Идентификаторы датасетов:

- `0`: BNCI2014001
- `1`: BNCI2014002
- `2`: Weibo2014

## Обучение

Запуск основного within-subject training pipeline:

```bash
python main.py --dataset 0 --model CUPY_SNN_LIF_READOUT --subject_id 1 --trial_num 1 --epoch 1500 --epoch2 600 --batch_size 128 --device 0
```

Запуск полного grid по выбранным вариантам:

```bash
python run_full_grid.py --dataset 0 --model CUPY_SNN_LIF_READOUT --epoch 1500 --epoch2 600 --batch_size 128 --device 0
```

Быстрый smoke grid:

```bash
python run_full_grid.py --smoke --subjects 1 --variants aug_base
```

## Примечания

Репозиторий организован под эксперименты для диссертационной работы, а не как отдельная pip-библиотека. Некоторые CuPy-based SNN модели требуют CUDA-окружение и подходящую сборку CuPy. CPU-only импорт baseline-моделей поддерживается через lazy loading в `src/models`.
