FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV BOT_TOKEN=default_token_if_any

CMD ["python", "bot/bot.py"]