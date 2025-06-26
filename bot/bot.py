from telegram.ext import ApplicationBuilder, MessageHandler, CommandHandler, ContextTypes, filters, CallbackQueryHandler
from telegram import Update, ReplyKeyboardMarkup, KeyboardButton
from PIL import Image
from dotenv import load_dotenv
import onnxruntime as ort
import numpy as np
import io
import os
import logging


logging.basicConfig(
    filename='bot.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


load_dotenv()
TOKEN = os.getenv("BOT_TOKEN")


session = ort.InferenceSession("/Users/yurishcherbakov/Desktop/garbage-classifier-bot/bot/trashnet.onnx")
classes = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]


menu_keyboard = ReplyKeyboardMarkup(
    keyboard=[
        [KeyboardButton("📄 Описание"), KeyboardButton("📬 Контакты")]
    ],
    resize_keyboard=True
)

def preprocess_image(image):
    image = image.resize((64, 64))
    image = np.array(image).astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0)
    return image


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    photo = update.message.photo[-1]
    file = await photo.get_file()
    bio = await file.download_as_bytearray()
    
    image = Image.open(io.BytesIO(bio)).convert("RGB")
    input_tensor = preprocess_image(image)
    outputs = session.run(None, {"input": input_tensor})
    
    logger.info(f"outputs: {outputs}")
    
    predicted_index = int(np.argmax(outputs[0]))
    predicted_class = classes[predicted_index]
    
    await update.message.reply_text(
        f"🧠 Обнаружен мусор: *{predicted_class}*",
        parse_mode="Markdown",
        reply_markup=menu_keyboard
    )


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 Привет! Я бот для распознавания мусора. Отправь мне фотографию, и я подскажу, что на ней.",
        reply_markup=menu_keyboard
    )


async def handle_menu_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text
    if text == "📄 Описание":
        await update.message.reply_text("📄 Я классифицирую мусор по фото: картон, стекло, металл, бумага, пластик или прочее.")
    elif text == "📬 Контакты":
        await update.message.reply_text("📬 Контакты разработчика: @yourusername\nGitHub: https://github.com/yourprofile")


def main():
    logger.info("Запуск Telegram-бота...")
    app = ApplicationBuilder().token(TOKEN).build()
    
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_menu_command))
    
    app.run_polling()

if __name__ == "__main__":
    main()
