from rest_framework import serializers
from django.db import transaction

from app.models import User
from app.api.v1.features.phone_cnn_service import is_phone_bool


def update_user_logic(validated_data, instance=None):
    data = validated_data.copy()

    password = data.pop("password", None)
    phone = data.pop("phone", None)

    if instance is None:
        user = User(**data)
    else:
        user = instance
        for key, value in data.items():
            setattr(user, key, value)

    if phone is not None:
        if is_phone_bool(phone):
            user.phone = phone
        else:
            raise serializers.ValidationError({
                "phone": "Вы ввели неправильный номер телефона"
            })
    if password:
        user.set_password(password)
    else:
        if instance is None:
            user.set_unusable_password()

    try:
        with transaction.atomic():
            user.save()
    except Exception:
        raise serializers.ValidationError({"non_field_errors": "Не удалось сохранить пользователя"})

    return user


class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = [
            "id",
            "username",
            "password",
            "phone",
            "email",
            "telegram_username",
            "telegram_chat_id",
        ]
        extra_kwargs = {
            "password": {"write_only": True},
            "email": {"required": False, "allow_blank": True},
            "phone": {"required": False, "allow_blank": True},
        }

    def create(self, validated_data):
        try:
            user = update_user_logic(validated_data, instance=None)
            return user
        except serializers.ValidationError as e:
            raise e
        except Exception as e:
            raise serializers.ValidationError(e)

    def update(self, instance, validated_data):
        try:
            user = update_user_logic(validated_data, instance=instance)
            return user
        except serializers.ValidationError as e:
            raise e
        except Exception as e:
            raise serializers.ValidationError(e)


class UserCreateUpdateSerializer(serializers.ModelSerializer):
    password = serializers.CharField(
        required=False,
        allow_blank=True,
        write_only=True
    )

    class Meta:
        model = User
        fields = [
            "username",
            "password",
            "phone",
            "email",
            "telegram_username",
            "telegram_chat_id",
        ]

    def update(self, instance: User, validated_data):
        try:
            user = update_user_logic(validated_data, instance=instance)
            return user
        except serializers.ValidationError as e:
            raise e
        except Exception as e:
            raise serializers.ValidationError(e)

    def create(self, validated_data):
        try:
            user = update_user_logic(validated_data, instance=None)
            return user
        except serializers.ValidationError as e:
            raise e
        except Exception as e:
            raise serializers.ValidationError(e)

