from cryptography.fernet import Fernet
from django.db import models
from django.contrib.auth.models import AbstractUser
from django.contrib.postgres.fields import ArrayField
from solo.models import SingletonModel
from dotenv import load_dotenv
import os

load_dotenv(encoding="utf-8")


class User(AbstractUser):
    email = models.EmailField(unique=True, null=True, blank=True)
    phone = models.CharField(max_length=20, null=True, blank=True)
    telegram_username = models.CharField(max_length=255, null=True, blank=True)
    telegram_chat_id = models.CharField(max_length=255, null=True, blank=True)

    notify_email = models.BooleanField(default=True)
    notify_sms = models.BooleanField(default=False)
    notify_telegram = models.BooleanField(default=False)

    class Meta:
        verbose_name = "Пользователь"
        verbose_name_plural = "Пользователи"

