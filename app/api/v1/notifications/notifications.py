from django.core.exceptions import ObjectDoesNotExist
from rest_framework.views import APIView
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework import status
from drf_spectacular.utils import extend_schema, OpenApiParameter
import logging
from django.conf import settings
from django.contrib.auth import get_user_model

from app.api.v1.celery_tasks import (
    send_email_task, send_telegram_task, send_sms_task, send_notification_fallback_task
)

logger = logging.getLogger(__name__)

User = get_user_model()


class SendNotificationView(APIView):
    permission_classes = [IsAuthenticated]

    @extend_schema(
        tags=['Notifications'],
        parameters=[
            OpenApiParameter(
                name="message",
                description="Ваше сообщение",
                type=str,
                required=True
            )
        ],
        responses={
            202: {"description": "Задача отправки уведомления поставлена в очередь"},
            400: {"description": "Отсутствует текст сообщения"},
            404: {"description": "Пользователь не найден"},
        }
    )
    def post(self, request, user_id):
        """
        Асинхронная отправка уведомления: \n
        Email -> SMS -> Telegram
        """
        try:
            user_to_send = User.objects.get(id=user_id)
        except ObjectDoesNotExist:
            return Response(
                {"error": f"Пользователь с id={user_id} не найден"},
                status=status.HTTP_404_NOT_FOUND
            )

        message = request.query_params.get("message")
        if not message:
            return Response(
                {"error": "Вы не передали текст уведомления"},
                status=status.HTTP_400_BAD_REQUEST
            )

        send_notification_fallback_task.delay(user_id, message)

        return Response(
            {"status": "Задача отправки уведомления запущена"},
            status=status.HTTP_202_ACCEPTED
        )


class SendEmailNotificationView(APIView):
    permission_classes = [IsAuthenticated]

    @extend_schema(
        tags=['Notifications (Debug)'],
        parameters=[
            OpenApiParameter(
                name="message",
                description="Ваше сообщение",
                type=str,
                required=True
            )
        ],
        responses={
            202: {"description": "Задача отправки уведомления поставлена в очередь"},
            400: {"description": "Отсутствует текст сообщения"},
            404: {"description": "Пользователь не найден"},
        }
    )
    def post(self, request, user_id):
        """
        Асинхронная отправка уведомления на почту: \n
        (Для теста)
        """
        try:
            user_to_send = User.objects.get(id=user_id)
        except ObjectDoesNotExist:
            return Response(
                {"error": f"Пользователь с id={user_id} не найден"},
                status=status.HTTP_404_NOT_FOUND
            )

        message = request.query_params.get("message")
        if not message:
            return Response(
                {"error": "Вы не передали текст уведомления"},
                status=status.HTTP_400_BAD_REQUEST
            )

        send_email_task.delay(
            message=message,
            username=user_to_send.username,
            recipient=user_to_send.email,
        )

        return Response(
            {"status": "Задача отправки уведомления запущена"},
            status=status.HTTP_202_ACCEPTED
        )


class SendSmsNotificationView(APIView):
    permission_classes = [IsAuthenticated]

    @extend_schema(
        tags=['Notifications (Debug)'],
        parameters=[
            OpenApiParameter(
                name="message",
                description="Ваше сообщение",
                type=str,
                required=True
            )
        ],
        responses={
            202: {"description": "Задача отправки уведомления поставлена в очередь"},
            400: {"description": "Отсутствует текст сообщения"},
            404: {"description": "Пользователь не найден"},
        }
    )
    def post(self, request, user_id):
        """
        Асинхронная отправка SMS-уведомления: \n
        (Для теста)
        """
        try:
            user_to_send = User.objects.get(id=user_id)
        except User.DoesNotExist:
            return Response(
                {"error": f"Пользователь с id={user_id} не найден"},
                status=status.HTTP_404_NOT_FOUND
            )

        # Проверяем наличие номера телефона
        if not user_to_send.phone or not user_to_send.phone.strip().startswith("+"):
            return Response(
                {
                    "error": "Номер телефона пользователя не указан или имеет неверный формат "
                             "(должен начинаться с '+')"
                },
                status=status.HTTP_400_BAD_REQUEST
            )

        message = request.query_params.get("message")
        if not message:
            return Response(
                {"error": "Вы не передали текст уведомления"},
                status=status.HTTP_400_BAD_REQUEST
            )

        send_sms_task.delay(
            api_id=settings.API_ID,
            to=user_to_send.phone,
            message=message,
            sms_from=settings.SMS_FROM,
        )

        return Response(
            {"status": "Задача отправки SMS запущена"},
            status=status.HTTP_202_ACCEPTED
        )


class SendTelegramNotificationView(APIView):
    permission_classes = [IsAuthenticated]

    @extend_schema(
        tags=['Notifications (Debug)'],
        parameters=[
            OpenApiParameter(
                name="message",
                description="Ваше сообщение",
                type=str,
                required=True
            )
        ],
        responses={
            202: {"description": "Задача отправки уведомления поставлена в очередь"},
            400: {"description": "Отсутствует текст сообщения"},
            404: {"description": "Пользователь не найден"},
        }
    )
    def post(self, request, user_id):
        """
        Асинхронная отправка уведомления в Telegram: \n
        (Для теста)
        """
        try:
            user_to_send = User.objects.get(id=user_id)
        except User.DoesNotExist:
            return Response(
                {"error": f"Пользователь с id={user_id} не найден"},
                status=status.HTTP_404_NOT_FOUND
            )

        if not user_to_send.telegram_chat_id:
            return Response(
                {"error": "Telegram ID пользователя не указан"},
                status=status.HTTP_400_BAD_REQUEST
            )

        message = request.query_params.get("message")
        if not message:
            return Response(
                {"error": "Вы не передали текст уведомления"},
                status=status.HTTP_400_BAD_REQUEST
            )

        send_telegram_task.delay(
            token=settings.BOT_TOKEN,
            chat_id=user_to_send.telegram_chat_id,
            text=message,
        )

        return Response(
            {"status": "Задача отправки в Telegram запущена"},
            status=status.HTTP_202_ACCEPTED
        )



