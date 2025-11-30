from django.db import models
from django.contrib.auth.models import AbstractUser


class User(AbstractUser):
    email = models.EmailField(unique=True, null=True, blank=True)
    phone = models.CharField(max_length=32, null=True, blank=True)
    telegram_username = models.CharField(max_length=255, null=True, blank=True)
    telegram_chat_id = models.CharField(max_length=255, null=True, blank=True)

    class Meta:
        verbose_name = "Пользователь"
        verbose_name_plural = "Пользователи"

