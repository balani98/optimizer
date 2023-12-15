from django.urls import path
from . import views

urlpatterns = [
    path("", views.explorer_home_page, name="home"),
    path("userguide", views.explorer_user_guide, name="explorer_userguide"),
    path("userguide/goalseek", views.goalseek_user_guide, name="goalseek_userguide"),
    path("userguide/predictor", views.predictor_user_guide, name="predictor_userguide"),
    path("userguide/optimizer", views.optimizer_user_guide, name="optimizer_userguide"),
    path(
        "ajax/download_sample_csv/",
        views.download_sample_csv,
        name="download_sample_csv",
    ),
    # path("", views.explorer_upload_csv, name='explorer_upload_csv'),
    path(
        "ajax/is_weekly_selected/", views.is_weekly_selected, name="is_weekly_selected"
    ),
      path(
        "ajax/is_convert_to_weekly_data/", views.convert_to_weekly_data, name="convert_to_weekly_data"
    ),
    path("ajax/date_check/", views.date_check, name="date_check"),
    path("ajax/dimension_check/", views.dimension_check, name="dimension_check"),
    path("ajax/spent_check/", views.spent_check, name="spent_check"),
    path("ajax/target_check/", views.target_check, name="target_check"),
    path("ajax/target_type_check/", views.target_type_check, name="target_type_check"),
    path("ajax/dimension_grouping_check", views.dimension_grouping_check, name="dimension_grouping_check"),
    path("ajax/cpm_check/", views.cpm_check, name="cpm_check"),
    path("ajax/chart_filter/", views.chart_filter, name="chart_filter"),
    # path("ajax/bar_chart_filter/", views.bar_chart_filter, name='bar_chart_filter'),
    # path("sample/", views.sample, name='sample'),
]