# Notification backend
# Backend / nginx
### Django server
Для debug:
```shell
poetry run python manage.py runserver 8000
```
Для production:
```shell
gunicorn app_project.wsgi:application --bind 127.0.0.1:8000 --workers 3
```
### Celery
```shell
set PYTHONPATH=C:\Users\your_user\PycharmProjects\work_backend\backend
cd backend
celery -A celery_config:app worker -l info
```
Или (для debug)
```shell
celery -A celery_config:app worker -l info -P solo
celery -A celery_config:app worker -l info --pool=solo
```
---
### Telegram bot
```shell
poetry run python manage.py run_telegram_bot
```
