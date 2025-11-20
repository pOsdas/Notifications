from django.core.mail import send_mail
from django.conf import settings
import logging

logger = logging.getLogger(__name__)


def send_email(
        username: str,
        message: str,
        recipient: str,
) -> bool:
    subject = f"Тестовое уведомление на почту"

    try:
        send_mail(
            subject=subject,
            message=message,
            from_email=settings.DEFAULT_FROM_EMAIL,
            recipient_list=[recipient],
            fail_silently=False,
        )
        logger.info("Успешно отправлено письмо пользователю: %s", username)
        return True

    except Exception as e:
        logger.exception("Ошибка при отправке письма: %s", e)
        return False
