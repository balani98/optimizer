from django.urls import path
from . import views

urlpatterns = [
    path("", views.optimizer_home_page, name="optimizer_home_page"),
    path(
        "ajax/left_pannel_dimension_min_max/",
        views.dimension_min_max,
        name="dimension_min_max",
    ),
    path(
        "ajax/optimizer_save_the_plan/",
        views.optimizer_save_the_plan,
        name="optimizer_save_the_plan",
    ),
]