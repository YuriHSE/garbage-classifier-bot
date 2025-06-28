# 🗑️ Garbage Classifier Bot

[![Docker Pulls](https://img.shields.io/docker/pulls/yurihse/garbage-classifier-bot)](https://hub.docker.com/r/yurihse/garbage-classifier-bot)
[![Python Version](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/)

Telegram-бот для классификации мусора с использованием ML. Готов к развертыванию через Docker.

## Описание модели

Подробное описание модели и принципов её работы можно найти в [MODEL_DESCRIPTION.md](MODEL_DESCRIPTION.md).

## 🚀 Быстрый старт

### Предварительные требования
- Установленный [Docker](https://www.docker.com/get-started)
- Токен бота от [@BotFather](https://t.me/BotFather)

### Запуск через Docker (рекомендуется)
```bash
docker run -e "BOT_TOKEN=your_token_here" yurihse/garbage-classifier-bot
```

# 🛠 Локальная установка

## 1. Клонирование репозитория

```bash
git clone https://github.com/e/garbage-classifier-bot.git
cd garbage-classifier-bot
```

## 2. Настройка окружения

Создайте файл `.env` со следующим содержимым:

```bash
echo "BOT_TOKEN=your_token_here" > .env
```

## 3. Запуск через Docker Compose

```bash
docker-compose up -d
```

# 🌍 Поддерживаемые платформы

Бот поддерживает работу на:

- ✅ **Linux** (amd64/arm64)
- ✅ **Windows** (требуется WSL2)
- ✅ **macOS** (Intel / Apple Silicon)

# 🔧 Устранение неполадок

## Проблема: Platform warning на Windows

**Сообщение об ошибке:**

```bash
WARNING: The requested image's platform (linux/arm64) does not match...
```

**Комментарий:**

Это предупреждение связано с несовпадением архитектур (например, `linux/arm64` и `windows/amd64`).  
Обычно это **не мешает работе контейнера**, и бот продолжает функционировать корректно.

Если вы хотите избавиться от предупреждения, можно явно указать платформу при загрузке образа:

```bash
docker pull --platform linux/amd64 yurihse/garbage-classifier-bot
```

> ⚠️ **Примечание:** В большинстве случаев это предупреждение можно игнорировать.
