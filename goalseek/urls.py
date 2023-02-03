from django.urls import path
from . import views

urlpatterns = [
    path("", views.goalseek_home_page, name="goalseek_home_page"),
    path("ajax/conversion_bound/", views.conversion_range, name="get_conversion_range"),
    path("ajax/left_panel_submit/", views.left_panel_submit, name="left_panel_submit")
]