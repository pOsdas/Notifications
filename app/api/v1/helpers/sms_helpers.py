import logging
import httpx

logger = logging.getLogger(__name__)


def send_sms(
        api_id: str,
        to: str,
        message: str,
        sms_from: str,
) -> tuple[bool, str]:
    if not to.startswith("+"):
        return False, f"Неверный формат номера: {to}"

    url = "https://sms.ru/sms/send"
    payload = {
        "api_id": api_id,
        "to": to,
        "msg": message,
        "from": sms_from,
        "json": 1,
    }
    try:
        with httpx.Client() as client:
            resp = client.post(url, data=payload, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            if data.get("status") != "OK":
                reason = data.get("status_text", "Неизвестная error")
                logger.error(f"SMS.ru ответил с ошибкой: {reason}")
                return False, f"API ошибка: {reason}"

            sms_info = data.get("sms", {}).get(to)
            if not sms_info:
                return False, "Неправильная SMS структура ответа"

            sms_status = sms_info.get("status")
            sms_text = sms_info.get("status_text", "")

            if sms_status == "OK":
                logger.info(f"SMS успешно отправлено to={to}")
                return True, "OK"

            logger.error(f"SMS ошибка to {to}: {sms_text}")
            return False, sms_text
    except Exception as e:
        logger.exception("sms.ru ошибка сети: %s", e)
        return False, f"Ошибка сети: {e}"
