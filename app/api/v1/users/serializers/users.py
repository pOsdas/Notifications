from rest_framework import serializers

from app.models import User


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
            "notify_email",
            "notify_sms",
            "notify_telegram",
        ]
        extra_kwargs = {
            "password": {"write_only": True},
            "email": {"required": False, "allow_blank": True},
            "phone": {"required": False, "allow_blank": True},
        }

    def create(self, validated_data):
        password = validated_data.pop("password", None)
        user = User(**validated_data)
        if password:
            user.set_password(password)
        else:
            user.set_unusable_password()
        user.save()
        return user

    def update(self, instance, validated_data):
        password = validated_data.pop("password", None)
        for attr, value in validated_data.items():
            setattr(instance, attr, value)
        if password:
            instance.set_password(password)
        instance.save()
        return instance


class UserCreateUpdateSerializer(serializers.ModelSerializer):
    password = serializers.CharField(
        required=False,
        allow_blank=True,
        write_only=True
    )

    notify_email = serializers.BooleanField(required=False, default=True)
    notify_sms = serializers.BooleanField(required=False, default=False)
    notify_telegram = serializers.BooleanField(required=False, default=False)

    class Meta:
        model = User
        fields = [
            "username",
            "password",
            "phone",
            "email",
            "telegram_username",
            "telegram_chat_id",
            "notify_email",
            "notify_sms",
            "notify_telegram",
        ]

    def update(self, instance: User, validated_data):
        password = validated_data.pop("password", None)
        user = User(**validated_data)

        if password:
            instance.set_password(password)
        else:
            user.set_unusable_password()
        user.save()
        return user

    def create(self, validated_data):
        password = validated_data.pop("password", None)
        user = User(**validated_data)

        if password:
            user.set_password(password)
        else:
            user.set_unusable_password()

        user.save()
        return user

