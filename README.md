# Многослойный персептрон

Требования: Python 3.11, NumPy и Matplotlib.

```bash
python3.11 -m pip install numpy matplotlib
```

Данные: `task/data.csv`, без заголовка (ID, класс B/M, 30 числовых признаков).
Команды из корня проекта, по порядку:

```bash
python3.11 src/mlp.py analyze          # Анализ данных
python3.11 src/mlp.py split            # Разделение на train и validation
python3.11 src/mlp.py train            # Обучение с настройками из отчёта
python3.11 src/mlp.py predict          # Оценка сохранённой модели
python3.11 src/mlp.py --help           # Все параметры
```

По умолчанию seed разбиения и обучения равен 42. Результаты сохраняются в `results/`.
Для обучения без окна графиков добавьте `--no-show`.
