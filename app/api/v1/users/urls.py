from django.urls import path
from .users import UserInfoView, ChangeUserView, CreateUserView, DeleteUserView, UsersListView
urlpatterns = [
    # Пользователи
    path('get-user/<int:user_id>/', UserInfoView.as_view(), name='get-user'),
    path('get-users/', UsersListView.as_view(), name='get-users'),
    path('delete-user/<int:user_id>/', DeleteUserView.as_view(), name='delete-user'),
    path('change-user/<int:user_id>/', ChangeUserView.as_view(), name='change-user'),
    path('create-user/', CreateUserView.as_view(), name='create-user')
]
