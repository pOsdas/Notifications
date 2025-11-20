from django.contrib import admin
from .models import User
from django.contrib.auth.admin import UserAdmin


@admin.register(User)
class CustomUserAdmin(UserAdmin):
    fieldsets = (
        (None, {'fields': ('id', 'username', 'password', 'phone', 'email', 'telegram_username', 'telegram_chat_id')}),
        ('Персональные данные', {
            'fields': ('first_name', 'last_name'),
            'classes': ('collapse',),
        })
    )

    add_fieldsets = (
        (None, {'fields': ('id', 'username', 'password1', 'password2', 'phone', 'email', 'telegram_username', 'telegram_chat_id')}),
        ('Персональные данные', {
            'fields': ('first_name', 'last_name'),
            'classes': ('collapse',),
        }),
    )

    list_display = (
        'username', 'first_name', 'last_name', 'phone', 'email', 'telegram_username', 'telegram_chat_id'
    )
    readonly_fields = ('id',)