import re
from django.contrib.auth.decorators import login_required
import os
from django.http import HttpResponse, JsonResponse
from django.shortcuts import render
from django.contrib import messages
from datetime import datetime
import pandas as pd
import numpy as np
import json
import boto3 
from saved_plans.models import SavedPlan
from django.views.decorators.csrf import csrf_exempt
# from django.core.files.storage import FileSystemStorage
# import os
# from django.conf import settings
# import uuid

# custom imports
from .goalseek import (
    dimension_bound,
    conversion_bound,
    optimizer_conversion
)


ERROR_DICT = {
    "5002": "Value Error",
    "5003": "Type Error",
    "5004": "Incorrect Date Format",
}
ENVIRONMENT = os.getenv('ENVIRONMENT')
# for production environment
if ENVIRONMENT == 'production':
    UPLOAD_FOLDER = 'var/www/optimizer/data/'
# for test environment
elif ENVIRONMENT == 'test':
    UPLOAD_FOLDER = "data/"
else:
    UPLOAD_FOLDER = "data/"
TEMP_ERROR_DICT = {"4002": "Value Error"}

# Create your views here.
@login_required()
def goalseek_home_page(request):
    maximum_of_minimum_value = -100
    print("optimizer_home_page")
    context = {}
    # Get the required items from session
    # discarded_items = request.session["discarded_items"]
    _uuid = request.session.get("_uuid")
    is_weekly_selected = request.session.get("is_weekly_selected")
    convert_to_weekly_data = request.session.get("convert_to_weekly_data")
    seasonality_from_session = request.session.get("seasonality")
    drop_dimension_from_session = request.session.get("drop_dimension")
    if seasonality_from_session:
        seasonality = seasonality_from_session
    else:
        seasonality = 0

    file_exists = os.path.exists(
        UPLOAD_FOLDER + "df_predictor_page_latest_data_{}.pkl".format(_uuid)
    )
    # context["run_wondow_onload"] = 1 if file_exists else 0
    if not file_exists:
        return JsonResponse(
            {"error": "Predictor page data is not submitted"}, status=403
        )
    else:
        df_predictor_page_latest_data = pd.read_pickle(
            UPLOAD_FOLDER + "df_predictor_page_latest_data_{}.pkl".format(_uuid)
        )
        scatter_plot_df = pd.read_pickle(
            UPLOAD_FOLDER + "scatter_plot_df_{}.pkl".format(_uuid)
        )
        goalseek_left_pannel_data = dimension_bound(df_predictor_page_latest_data, scatter_plot_df)
        for (key, value) in goalseek_left_pannel_data.items():
            if value[3] > maximum_of_minimum_value :
                maximum_of_minimum_value = value[3]
        context["maximum_of_minimum_value"] = maximum_of_minimum_value 
        stringified_goalseek_left_pannel_data = json.dumps(goalseek_left_pannel_data)
        # Checking for CPM selection
        if request.session.get("cpm_checked") == "True":
            context["cpm_checked"] = 1
        else:
            context["cpm_checked"] = 0
        if is_weekly_selected:
            print(f"is_weekly_selected : {is_weekly_selected}")
            context["is_weekly_selected"] = int(is_weekly_selected)
        
        if convert_to_weekly_data:
            print(f"convert_to_weekly_data : {convert_to_weekly_data}")
            context["convert_to_weekly_data"] = int(convert_to_weekly_data)

        context["seasonality"] = seasonality
        context["drop_dimension_from_session"] = drop_dimension_from_session
        context["goalseek_left_pannel_data"] = goalseek_left_pannel_data
        context[
            "stringified_goalseek_left_pannel_data"
        ] = stringified_goalseek_left_pannel_data
    return render(request, "goalseek/goalseek.html", context)


def conversion_range(request):
    context = {}
    _uuid = request.session.get("_uuid")
    body = json.loads(request.body)
    df_predictor_page_latest_data = pd.read_pickle(
            UPLOAD_FOLDER + "df_predictor_page_latest_data_{}.pkl".format(_uuid)
        )
    scatter_plot_df = pd.read_pickle(
            UPLOAD_FOLDER + "scatter_plot_df_{}.pkl".format(_uuid)
        )
    dimension_min_max = json.loads(body["dimension_min_max"])
    selected_dimensions = body['selected_dimensions'].split(",")
    conv_bound = conversion_bound(df_predictor_page_latest_data, scatter_plot_df, dimension_min_max, selected_dimensions)
    context['conversion_bound'] = conv_bound
    print(conv_bound)
    return JsonResponse(context)


def left_panel_submit(request):
    try:
        print("dimension_min_max")
        # dimension_min_max = {
        #     "Audio": [0, 5000],
        #     "CTV": [0, 5000],
        #     "Display": [0, 50000],
        #     "Email": [0, 50000],
        #     "Local Radio": [0, 50000],
        #     "Local TV": [0, 50000],
        #     "National TV": [0, 50000],
        #     "Native": [0, 50000],
        #     "Paid Social": [0, 50000],
        #     "SEM": [0, 50000],
        #     "Video": [0, 50000],
        # }
        # number_of_days = 2
        # total_budget = 500000

        context = {}
        dict_donut_chart_data = {}
        dict_line_chart_data = {}
        start_date = None
        end_date = None
        number_of_days = None

        df_predictor_page_latest_data = pd.read_pickle(
            UPLOAD_FOLDER + "df_predictor_page_latest_data_{}.pkl".format(
                request.session.get("_uuid")
            )
        )
        scatter_plot_df = pd.read_pickle(
            UPLOAD_FOLDER + "scatter_plot_df_{}.pkl".format(
                request.session.get("_uuid")
            )
        )

        # Get from request object
        body = json.loads(request.body)
        seasonality = int(body["seasonality"])
        #selected_dimensions = request.GET.get("selectedDimensions").split(",") 
        selected_dimensions = (body['selected_dimensions']).split(",")
        total_conversion = int(body['total_conversion'])
        discarded_dimensions = json.loads(body["discarded_dimensions"])
        dimension_min_max = json.loads(body['dimension_min_max'])
        if seasonality:
            print(
                f"\ndimension_min_max - seasonality:{seasonality}, Running optimizer_with_seasonality_class"
            )
            start_date = datetime.strptime(
                body["start_date"], "%m-%d-%y"
            ).date()
            end_date = datetime.strptime(body["end_date"], "%m-%d-%y").date()
            print(start_date, end_date)
            date_range = [start_date, end_date]
        else:
            print(
                f"\ndimension_min_max - seasonality:{seasonality}, Running optimizer_class"
            )
            number_of_days = int(body["number_of_days"])
            optimize_con_obj = optimizer_conversion(df_predictor_page_latest_data)
            df_spend_dis = pd.DataFrame(request.session.get('df_spend_dis'))
            df_optimizer_results_post_min_max = pd.DataFrame()
            print('number_of_days',number_of_days)
            (df_optimizer_results_post_min_max,message) = optimize_con_obj.execute( scatter_plot_df, 
                                                                                    total_conversion, 
                                                                                    number_of_days, 
                                                                                    df_spend_dis, 
                                                                                    discarded_dimensions, 
                                                                                    dimension_min_max , 
                                                                                    selected_dimensions)
        print(

            "\n number_of_days",
            number_of_days,
            "\n start_date",
            start_date,
            "\n end_date",
            end_date,
            "\n total_conversion",
            total_conversion,
        )

        # Table1
        df_table_1_data = df_optimizer_results_post_min_max[
            [
                'dimension', 
                'original_median_budget_per_day',
                'estimated_budget_per_day', 
                'buget_allocation_old_%',
                'buget_allocation_new_%', 
                'estimated_budget_for_n_days',
                'recommended_return_per_day', 
                'recommended_return_%',
                'recommended_return_for_n_days'
            ]
        ]

        # Total values
        df_sum_ = df_table_1_data.sum()
        df_sum_['original_median_budget_per_day'] = df_sum_['original_median_budget_per_day'].round()
        df_sum_['estimated_budget_per_day'] = df_sum_['estimated_budget_per_day'].round()
        df_sum_['buget_allocation_old_%'] = df_sum_['buget_allocation_old_%'].round()
        df_sum_['buget_allocation_new_%'] = df_sum_['buget_allocation_new_%'].round()
        df_sum_['estimated_budget_for_n_days'] = df_sum_['estimated_budget_for_n_days']
        df_sum_[df_sum_.index == "dimension"] = "Total"
        df_table_1_data['original_median_budget_per_day'] = df_table_1_data['original_median_budget_per_day'].round()
        df_table_1_data['estimated_budget_per_day'] = df_table_1_data['estimated_budget_per_day'].round()
        df_table_1_data['buget_allocation_old_%'] = df_table_1_data['buget_allocation_old_%'].round(decimals=2)
        df_table_1_data['buget_allocation_new_%'] = df_table_1_data['buget_allocation_new_%'].round(decimals=2)
        df_table_1_data['estimated_budget_for_n_days'] = df_table_1_data['estimated_budget_for_n_days'].round()
        df_table_1_data = df_table_1_data.append(df_sum_, ignore_index=True)
        df_table_1_data = df_table_1_data
        is_weekly_selected = request.session.get("is_weekly_selected")
        convert_to_weekly_data = request.session.get("convert_to_weekly_data")
        df_table_for_csv = pd.DataFrame()
        if is_weekly_selected or convert_to_weekly_data:
            df_table_for_csv =  df_table_1_data.rename(columns = {'original_median_budget_per_day':'original_median_budget_per_week',
                                           'estimated_budget_per_day':'estimated_budget_per_week',
                                           'estimated_budget_for_n_days':'estimated_budget_for_n_weeks',
                                           'recommended_return_per_day':'recommended_return_per_week',
                                           'recommended_return_for_n_days':'recommended_return_for_n_weeks'
                                           }, inplace = False)
        else:
            df_table_for_csv = df_table_1_data
        #   To Download CSV
        # df_optimizer_results_post_min_max.to_csv("optimizer_results_post_min_max.csv")
        csv_optimizer_download_csv_data = df_table_for_csv.to_csv()
        json_dumped_optimizer_download_csv = json.dumps(
            csv_optimizer_download_csv_data, separators=(",", ":")
        )

        # Bar Chart
        # df_donut_chart_data = df_optimizer_results_post_min_max[
        #     ["dimension", "recommended_budget_per_day"]
        # ]
        dict_donut_chart_data["dimension"] = df_optimizer_results_post_min_max[
            "dimension"
        ].tolist()
        dict_donut_chart_data[
            "buget_allocation_old_%"
        ] = df_optimizer_results_post_min_max["buget_allocation_old_%"].tolist()
        dict_donut_chart_data[
            "buget_allocation_new_%"
        ] = df_optimizer_results_post_min_max["buget_allocation_new_%"].tolist()

        json_table_1_data = df_table_1_data.to_dict("records")
        print("json_table_1_data", json_table_1_data)
        # print("json_donut_chart_data", json_donut_chart_data)
        print("dict_donut_chart_data", dict_donut_chart_data)
        # Table1
        context["optimizer_download_csv_json"] = json_dumped_optimizer_download_csv
        context["json_table_1_data"] = json_table_1_data
        context["dict_donut_chart_data"] = dict_donut_chart_data
        return JsonResponse(context)
    except Exception as e:
        return HttpResponse(ERROR_DICT[str(e)], status=403)
        # messages.error(request, e)
        # return JsonResponse({"error": str(e)}, status=403)