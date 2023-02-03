from django.urls import path
from . import views

urlpatterns = [
    path("list/", views.saved_plan_list, name="saved_plan_list"),
    path('ajax/delete', views.delete_saveplan, name="delete_save_plan")
]
