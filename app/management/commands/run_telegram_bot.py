from django.core.management.base import BaseCommand
from app.api.v1.helpers.telegram_helpers import build_bot


class Command(BaseCommand):
    help = "Запускает Telegram бота"

    def handle(self, *args, **options):
        bot = build_bot()
        self.stdout.write(self.style.SUCCESS("Telegram bot запущен"))
        bot.run_polling()
