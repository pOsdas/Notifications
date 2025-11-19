# CRUD пользователей
from rest_framework import status, generics
from rest_framework.views import APIView
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from drf_spectacular.utils import extend_schema

from app.models import User
from app.api.v1.users.serializers import UserSerializer, UserCreateUpdateSerializer


class UserInfoView(generics.RetrieveAPIView):
    queryset = User.objects.all()
    serializer_class = UserSerializer
    permission_classes = [IsAuthenticated]
    lookup_field = "id"

    @extend_schema(tags=['Users'])
    def get(self, request, *args, **kwargs):
        """Получить пользователя по id"""
        return super().get(request, *args, **kwargs)


class UsersListView(generics.ListAPIView):
    queryset = User.objects.all()
    serializer_class = UserSerializer
    permission_classes = [IsAuthenticated]

    @extend_schema(tags=['Users'])
    def get(self, request, *args, **kwargs):
        """Получить всех пользователей"""
        return super().get(request, *args, **kwargs)


class ChangeUserView(generics.UpdateAPIView):
    queryset = User.objects.all()
    serializer_class = UserCreateUpdateSerializer
    permission_classes = [IsAuthenticated]
    lookup_field = "id"

    @extend_schema(tags=['Users'])
    def patch(self, request, *args, **kwargs):
        """
        Изменение пользователя \n
        При тестировании через swagger необходимо убрать лишнее поля в примере `Schema`
        """
        return super().patch(request, *args, **kwargs)



class DeleteUserView(generics.DestroyAPIView):
    queryset = User.objects.all()
    permission_classes = [IsAuthenticated]
    lookup_field = "id"

    @extend_schema(tags=['Users'])
    def delete(self, request, *args, **kwargs):
        """Удалить пользователя"""
        user = request.user
        user_to_delete = self.get_object()

        if user.id == user_to_delete.id:
            return Response(
                {"error": "Нельзя удалить самого себя"},
                status=status.HTTP_400_BAD_REQUEST
            )

        return super().delete(request, *args, **kwargs)


class CreateUserView(generics.CreateAPIView):
    queryset = User.objects.all()
    serializer_class = UserCreateUpdateSerializer
    permission_classes = [IsAuthenticated]

    @extend_schema(tags=['Users'])
    def post(self, request, *args, **kwargs):
        """Создать пользователя"""
        return super().post(request, *args, **kwargs)
