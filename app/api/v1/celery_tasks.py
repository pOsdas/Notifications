import logging
from django.core.exceptions import ObjectDoesNotExist
from celery import shared_task
from django.conf import settings
from django.contrib.auth import get_user_model

from app.api.v1.helpers import send_sms, send_telegram_message, send_email

logger = logging.getLogger(__name__)


@shared_task(bind=True, max_retries=3)
def send_notification_fallback_task(
        self,
        user_id: int,
        message: str,
) -> bool:
    User = get_user_model()

    try:
        user = User.objects.get(id=user_id)
    except ObjectDoesNotExist:
        logger.error(f"Пользователь с id={user_id} не найден для отправки уведомления")
        return False

    # Email
    if user.email:
        try:
            if send_email(
                    message=message,
                    username=user.username,
                    recipient=user.email
            ):
                logger.info(f"Уведомление отправлено на email для user_id={user_id}")
                return True
        except Exception as e:
            logger.error(f"Ошибка при отправке email пользователю={user_id}: {str(e)}")

    # SMS
    if user.phone and all([settings.API_ID, settings.SMS_FROM]):
        try:
            if send_sms(
                    api_id=settings.API_ID,
                    to=user.phone,
                    message=message,
                    sms_from=settings.SMS_FROM,
            ):
                logger.info(f"Уведомление отправлено по SMS для user_id={user_id}")
                return True
        except Exception as e:
            logger.error(f"Ошибка при отправке sms пользователю={user_id}: {str(e)}")

    # Telegram
    if user.telegram_chat_id and settings.BOT_TOKEN:
        try:
            if send_telegram_message(
                    token=settings.BOT_TOKEN,
                    chat_id=user.telegram_chat_id,
                    text=message
            ):
                logger.info(f"Уведомление отправлено в Telegram для user_id={user_id}")
                return True
        except Exception as e:
            logger.error(f"Ошибка при отправке сообщения в telegram пользователю={user_id}: {str(e)}")

    logger.error(
        "Все методы отправки уведомления провалились для user_id: %s", user_id
    )

    if self.request.retries < self.max_retries:
        raise self.retry(countdown=60)

    return False


# --- Для debug ---
@shared_task(bind=True, max_retries=3)
def send_email_task(
        self,
        message: str,
        username: str,
        recipient: str
) -> bool:
    """
    Отправляет письмо пользователю
    """
    logger.debug(f"Task 'send_email_task' started: username={username}")

    try:
        send_email(
            message=message,
            username=username,
            recipient=recipient,
        )

        return True

    except Exception as e:
        logger.error("Ошибка при выполнении send_email_task: %s", e)
        raise self.retry(exc=e, countdown=60)


@shared_task(bind=True, max_retries=3)
def send_sms_task(
        self,
        api_id: str,
        to: str,
        message: str,
        sms_from: str,
) -> bool:
    """
    Отправляет sms пользователю
    """
    logger.debug(f"Task 'send_sms_task' started")

    try:
        send_sms(
            api_id=api_id,
            to=to,
            message=message,
            sms_from=sms_from,
        )
        return True

    except Exception as e:
        logger.error("Ошибка при выполнении send_sms_task: %s", e)
        raise self.retry(exc=e, countdown=60)


@shared_task(bind=True, max_retries=3)
def send_telegram_task(
        self,
        token: str,
        chat_id: int,
        text: str,
) -> bool:
    """
    Отправляет уведомление пользователю в telegram
    """
    logger.debug(f"Task 'send_telegram_task' started")

    try:
        ok = send_telegram_message(
            token=token,
            chat_id=chat_id,
            text=text,
        )
        if ok:
            return True

        raise Exception("Telegram API не вернул ok или произошла ошибка сети")

    except Exception as e:
        logger.error("Ошибка при выполнении send_telegram_task: %s", e)
        raise self.retry(exc=e, countdown=60)




