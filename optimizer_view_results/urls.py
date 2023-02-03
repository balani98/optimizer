from django.urls import path
from . import views

urlpatterns = [
    path("", views.optimizer_view_results, name="optimizer_view_results"),
    path("ajax/table_and_chart_results/", views.table_and_chart_results, name="table_and_chart_results")
]
