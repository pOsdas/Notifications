from django.apps import AppConfig


class BackendAppConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'app'

    def ready(self) -> None:
        """
        Поднимем CharCNN при старте с проверкой
        """
        from app.api.v1.features import phone_cnn_v2, phone_cnn_service
        from django.conf import settings

        ARTIFACT_PREFIX = settings.CHAR_CNN_MODEL_PREFIX

        try:
            model = phone_cnn_v2.load_artifacts(prefix=ARTIFACT_PREFIX)
        except Exception as e:
            raise RuntimeError(f"Не удалось загрузить CharCNN модель (prefix={ARTIFACT_PREFIX}): {e}") from e

        phone_cnn_service.MODEL = model
        print("CharCNN модель успешно загружена и готова к использованию")