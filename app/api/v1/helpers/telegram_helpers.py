import logging
import httpx
from telegram.ext import Application, CommandHandler
from asgiref.sync import sync_to_async
from django.conf import settings
from django.contrib.auth import get_user_model

logger = logging.getLogger(__name__)

User = get_user_model()


async def start(update, context):
    try:
        chat_id = update.effective_chat.id
        parts = update.message.text.split()
        if len(parts) < 2:
            update.message.reply_text("Ошибка: отсутствует user_id")
            return

        try:
            user_id = int(parts[1])
        except ValueError:
            await update.message.reply_text("Некорректный user_id")
            return

        try:
            user = await sync_to_async(User.objects.get)(id=user_id)
        except User.DoesNotExist:
            await update.message.reply_text("Пользователь не найден в базе")
            return

        user.telegram_chat_id = chat_id
        await sync_to_async(user.save)(update_fields=['telegram_chat_id'])

        await update.message.reply_text("Telegram подключен")
    except Exception as e:
        update.message.reply_text(f"Ошибка: {e}")


def build_bot():
    app = Application.builder().token(settings.BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    return app


def send_telegram_message(
        token: str,
        chat_id: int,
        text: str
) -> bool:
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": int(chat_id),
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True
    }
    try:
        with httpx.Client() as client:
            resp = client.post(url, json=payload, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            if not data.get("ok"):
                logger.error("Telegram API вернул не ok: %s", data)
                return False

            message_id = data.get("result", {}).get("message_id")
            logger.info("Успешно отправлено уведомление пользователю: %s", chat_id)
            return True

    except Exception as e:
        logger.exception("Ошибка при отправке уведомления: %s", e)
        return False
