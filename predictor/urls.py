from django.urls import path
from . import views

urlpatterns = [
    path("", views.predictor_home_page, name="predictor_home_page"),
    path(
        "ajax/get_predictor_start_and_end_dates/",
        views.get_predictor_start_and_end_dates,
        name="get_predictor_start_and_end_dates",
    ),
    path(
        "ajax/left_panel_submit/",
        views.predictor_ajax_left_panel_submit,
        name="predictor_ajax_left_panel_submit",
    ),
    path(
        "ajax/date_dimension_onchange/",
        views.predictor_ajax_date_dimension_onchange,
        name="predictor_ajax_date_dimension_onchange",
    ),
    path(
        "ajax/y_axis_onchange/",
        views.predictor_ajax_y_axis_onchange,
        name="predictor_ajax_y_axis_onchange",
    ),
    path(
        "ajax/get_seasonality_from_session/",
        views.get_seasonality_from_session,
        name="get_seasonality_from_session",
    ),
    path(
        "ajax/predictor_send_value/",
        views.predictor_ajax_predictor_send_value,
        name="predictor_ajax_predictor_send_value",
    ),
    path(
        "ajax/predictor_discard/",
        views.predictor_ajax_predictor_discard,
        name="predictor_ajax_predictor_discard",
    ),
    path(
        "ajax/progress_bar_var/",
        views.get_progress_bar_dict,
        name="predictor_progress_bar_var",
    ),
    path(
        "predictor_window_on_load/",
        views.predictor_window_on_load,
        name="predictor_window_on_load",
    ),
    path(
        "ajax/is_weekly_selected/",
        views.get_is_weekly_selected,
        name="get_is_weekly_selected",
    ),
    path(
        "ajax/convert_to_weekly_data/",
        views.get_convert_to_weekly_data,
        name="get_convert_to_weekly_data",
    ),
     path('download/pdf/', views.download_predictor_curves_pdf, name='download_pdf'),
    # path("", views.predictor, name="predictor"),
    # path("save/", views.predictor_save, name="predictor_save"),
    # path("delete/", views.predictor_delete, name="predictor_delete"),
]