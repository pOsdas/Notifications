from django.urls import path
from .notifications import (
    SendNotificationView, SendSmsNotificationView, SendTelegramNotificationView, SendEmailNotificationView
)
urlpatterns = [
    # Пользователи
    path('prod/<int:user_id>/', SendNotificationView.as_view(), name='send-notification'),
    path('telegram/<int:user_id>/', SendTelegramNotificationView.as_view(), name='send-notification-telegram'),
    path('email/<int:user_id>/', SendEmailNotificationView.as_view(), name='send-notification-email'),
    path('sms/<int:user_id>/', SendSmsNotificationView.as_view(), name='send-notification-sms'),
]
