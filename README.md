# 1T_hackaton_swarm

# Ветки
- points_solution: для 15 огней траектория просчитана, дроны летят по ней. Для неизвестных очагов летим А*
- base_solution: базовое решение без просчитанных траекторий, с худшей физикой передвижения
- rl_solution: решение, основанное на модели actor-critic, не успели обучить

# Настройка окружения

- Создать виртуальное окружение
  ```bash
  python -m venv
  ```
  Или с помощью uv:
  ```bash
  uv venv
  ```
- Активировать виртуальное окружение
  + На Linux
    ```bash
    source venv/bin/activate
    ```
  + На Windows
    ```bash
    source venv/Scripts/activate
    ```
- Установить зависимости:
  ```bash
  pip install requirements.txt
  ```
  Или с помощью uv:
  ```bash
  uv sync
  ```
  Или
  ```bash
  uv add -r requirements.txt
  ```

- Постваить свой порт в файле `config.py` в переменную `POST`

# Запуск

  ```bash
  python main.py
  ```
  Или с помощью uv:
  ```bash
  uv run main.py
  ```

