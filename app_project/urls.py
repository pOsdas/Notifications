from django.contrib import admin
from django.urls import path, include
from rest_framework.response import Response
from app.config import pydantic_settings
from rest_framework.decorators import api_view
from drf_spectacular.views import SpectacularAPIView, SpectacularSwaggerView, SpectacularRedocView

API_PREFIX = pydantic_settings.api.prefix
API_V1_PREFIX = pydantic_settings.api.v1.prefix


@api_view(['GET'])
def root_hello(request):
    return Response({"message": "hello from our app"})


urlpatterns = [
    path("admin/", admin.site.urls),
    path('', root_hello, name='root'),
    path(f"{API_PREFIX}{API_V1_PREFIX}/", include("app.api.v1.urls")),

    # Маршруты для OpenAPI схемы и Swagger UI / Redoc
    path(f'{API_PREFIX}{API_V1_PREFIX}/schema/', SpectacularAPIView.as_view(), name='schema'),
    path(f'{API_PREFIX}{API_V1_PREFIX}/docs/', SpectacularSwaggerView.as_view(url_name='schema'), name='swagger-ui'),
    path(f'{API_PREFIX}{API_V1_PREFIX}/redoc/', SpectacularRedocView.as_view(url_name='schema'), name='redoc'),
]
