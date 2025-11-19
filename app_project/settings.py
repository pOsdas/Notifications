from pathlib import Path
import logging
import os
import logging.config
import environ
from app.config import pydantic_settings


# Корневая директория проекта
BASE_DIR = Path(__file__).resolve().parent.parent

env = environ.Env()

env.read_env(
    os.path.join(BASE_DIR, '.env')
)

DOCKER = env.bool("DOCKER", False)

# Основные настройки
SECRET_KEY = pydantic_settings.secret_key
DEBUG = pydantic_settings.debug
ALLOWED_HOSTS = env.str("ALLOWED_HOSTS", "127.0.0.1,localhost").split(",")

# Настройка базы данных
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql_psycopg2',
        'NAME': env.str('POSTGRES_DB'),
        'USER': env.str('POSTGRES_USER'),
        'PASSWORD': env.str('POSTGRES_PASSWORD'),
        'HOST': env.str('POSTGRES_HOST') if DOCKER else 'localhost',
        'PORT': env.str('POSTGRES_PORT')
    }
}

# DATABASES["default"]["ENGINE"] = "django.db.backends.postgresql_async"
DATABASES["default"]["ENGINE"] = "django.db.backends.postgresql"

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# SMS
TWILIO_FROM = env.str("TWILIO_FROM")
TWILIO_AUTH_TOKEN = env.str("TWILIO_AUTH_TOKEN")
TWILIO_ACCOUNT_SID = env.str("TWILIO_ACCOUNT_SID")

# Telegram
TELEGRAM_API = env.str("TELEGRAM_API")
BOT_TOKEN = env.str("BOT_TOKEN")

# Почта
EMAIL_BACKEND = "django.core.mail.backends.smtp.EmailBackend"
EMAIL_HOST = env.str("EMAIL_HOST")
EMAIL_PORT = env.int("EMAIL_PORT")
EMAIL_USE_SSL = env.bool("EMAIL_USE_SSL", 0)
EMAIL_USE_TLS = env.bool("EMAIL_USE_TLS", 1)
EMAIL_HOST_USER = env.str("EMAIL_HOST_USER")
EMAIL_HOST_PASSWORD = env.str("EMAIL_HOST_PASSWORD")
DEFAULT_FROM_EMAIL = env.str("EMAIL_HOST_USER")

# Настройка Redis
REDIS_HOST = env.str("REDIS_HOST", "localhost") if DOCKER else '127.0.0.1'
REDIS_PORT = env.str("REDIS_PORT", "6379")  # if DOCKER else '127.0.0.1:6379'
REDIS_DB = env.str("REDIS_DB", "1")
REDIS_DECODE_RESPONSES = True

# Настройка Celery
CELERY_BROKER_URL = env.str("CELERY_BROKER_URL")  # f"redis://{_REDIS_HOST}:{_REDIS_PORT}/{REDIS_DB}"
CELERY_RESULT_BACKEND = env.str("CELERY_BROKER_URL")  # f"redis://{_REDIS_HOST}:{_REDIS_PORT}/{REDIS_DB}"

# Установленные приложения
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'rest_framework.authtoken',
    'drf_spectacular',
    'rest_framework',
    'app',
]

MIDDLEWARE = [
    'whitenoise.middleware.WhiteNoiseMiddleware',
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
]

ROOT_URLCONF = 'app_project.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [BASE_DIR / 'templates'],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

# ASGI_APPLICATION = 'app_project.asgi.application'
WSGI_APPLICATION = 'app_project.wsgi.application'

# Статические файлы
STATIC_URL = '/static/'
STATIC_ROOT = BASE_DIR / 'static'

# (Опционально) Настройки Django REST Framework
REST_FRAMEWORK = {
    'DEFAULT_AUTHENTICATION_CLASSES': [
        "rest_framework.authentication.BasicAuthentication",
        "rest_framework.authentication.TokenAuthentication"
    ],
    'DEFAULT_PERMISSION_CLASSES': [
        "rest_framework.permissions.IsAuthenticated",
    ],
    'DEFAULT_SCHEMA_CLASS': 'drf_spectacular.openapi.AutoSchema',
}

AUTH_USER_MODEL = "app.User"

# Русский язык
LANGUAGE_CODE = 'ru'
USE_I18N = True
USE_L10N = True

AUTH_PASSWORD_VALIDATORS = [
    {
        "NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.CommonPasswordValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.NumericPasswordValidator",
    },
    {
        "NAME": "app.api.core.CustomPasswordValidator",
    },
]

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "colored": {
            "()": "colorlog.ColoredFormatter",
            "format": "%(log_color)s%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
            "log_colors": {
                "DEBUG": "cyan",
                "INFO": "white",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "bold_red",
            },
        },
        "json": {
            "()": "pythonjsonlogger.jsonlogger.JsonFormatter",
            "fmt": "%(levelname)s %(name)s %(message)s %(asctime)s",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
            "formatter": "colored" if DEBUG else "json",
            "level": "DEBUG",
        },
    },
    "root": {
        "handlers": ["console"],
        "level": "INFO",
    },
    "loggers": {
        "django": {
            "handlers": ["console"],
            "level": "INFO",
            "propagate": False,
        },
    },
}

logging.config.dictConfig(LOGGING)