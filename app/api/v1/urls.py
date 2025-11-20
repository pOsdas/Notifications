from django.urls import path, include

urlpatterns = [
    path("notifications/", include("app.api.v1.notifications.urls")),
    path("users/", include("app.api.v1.users.urls")),
]