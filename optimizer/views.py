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
from .optimizer_helper_functions import (
    dimension_bound,
    optimizer_iterative,
    optimizer_iterative_seasonality
)


ERROR_DICT = {
    "5002": "Value Error",
    "5003": "Type Error",
    "5004": "Incorrect Date Format",
}
# Declaring the environment variable
ENVIRONMENT = os.getenv('ENVIRONMENT')  or 'test'
# for production environment
if ENVIRONMENT == 'production':
    UPLOAD_FOLDER = 'var/www/optimizer/data/'
# for test environment
elif ENVIRONMENT == 'test':
    UPLOAD_FOLDER = "data/"
else:
    UPLOAD_FOLDER = "data/"
TEMP_ERROR_DICT = {"4002": "Value Error"}


@login_required
def optimizer_home_page(request):
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
        # discarded_items_array = discarded_items.split(",")
        # df_param = df_predictor_page_latest_data[
        #     ~df_predictor_page_latest_data["dimension"].isin(discarded_items_array)
        # ]
        dimension_data = request.session.get('dimension_data')
        (optimizer_left_pannel_data,
         grouped_optimizer_left_pannel_data,
         grouped_channel_dictionary,
         flag_to_show_grouped_dimensions) = dimension_bound(df_predictor_page_latest_data, dimension_data)
        stringified_optimizer_left_pannel_data = json.dumps(optimizer_left_pannel_data)
        
        print("optimizer_left_pannel_data", optimizer_left_pannel_data)
        print(
            "stringified_optimizer_left_pannel_data",
            stringified_optimizer_left_pannel_data,
        )  
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
        grouped_optimizer_left_pannel_data = {
                                                'Display': {
                                                'sub_channels' : ['Display_Affiliate' , 'Display_Retargeting' , 'Display_Audio'],
                                                 'constraints': [0, 1000]
                                                }
                                            }
        if flag_to_show_grouped_dimensions == 1:
            context['grouped_optimizer_left_pannel_data'] = grouped_optimizer_left_pannel_data
          
            stringified_grouped_optimizer_left_pannel_data = json.dumps(grouped_optimizer_left_pannel_data)
            context['stringified_grouped_optimizer_left_pannel_data'] = stringified_grouped_optimizer_left_pannel_data
        context['flag_to_show_grouped_dimensions'] = flag_to_show_grouped_dimensions
        context["seasonality"] = seasonality
        context["drop_dimension_from_session"] = drop_dimension_from_session
        context["optimizer_left_pannel_data"] = optimizer_left_pannel_data
        context[
            "stringified_optimizer_left_pannel_data"
        ] = stringified_optimizer_left_pannel_data
        return render(request, "optimizer/optimizer_home_page.html", context)


def dimension_min_max(request):
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
            UPLOAD_FOLDER + "scatter_plot_df_{}.pkl".format(request.session.get("_uuid"))
        )

        # Get from request object
        body = json.loads(request.body)
        seasonality = int(body["seasonality"])
        dimension_min_max = json.loads(body["dimension_min_max"])
        total_budget = int(body["total_budget"])
        discarded_dimensions = json.loads(body["discarded_dimensions"])
        # cpm_checked = request.session.get('cpm_checked')
        if seasonality:
            print(
                f"\ndimension_min_max - seasonality:{seasonality}, Running optimizer_with_seasonality_class"
            )
            start_date = datetime.strptime(
                body["start_date"], "%m-%d-%y"
            ).date()
            end_date = datetime.strptime(body["end_date"], "%m-%d-%y").date()
            df_spend_dis = pd.DataFrame(request.session.get('df_spend_dis'))
            print(start_date, end_date)
            date_range = [start_date, end_date]
            optimizer_object = optimizer_iterative_seasonality(
                df_predictor_page_latest_data
            )
            try:
                (
                    df_optimizer_results_post_min_max
                ) = optimizer_object.execute(scatter_plot_df, total_budget, date_range, df_spend_dis, discarded_dimensions, dimension_min_max)
            except Exception as error:
                return JsonResponse({"error": str(error)}, status=501)
        else:
            print(
                f"\ndimension_min_max - seasonality:{seasonality}, Running optimizer_class"
            )
            number_of_days = int(body["number_of_days"])
            df_spend_dis = pd.DataFrame(request.session.get('df_spend_dis'))
            optimizer_object = optimizer_iterative(df_predictor_page_latest_data)
            try:
                 
                (
                    df_optimizer_results_post_min_max
                ) = optimizer_object.execute(
                    scatter_plot_df, total_budget, number_of_days, df_spend_dis, discarded_dimensions, dimension_min_max
                )
            except Exception as error:
                return JsonResponse({"error": str(error)}, status=501)
        df_optimizer_results_post_min_max = df_optimizer_results_post_min_max.round(2)

        print(
            "\n dimension_min_max",
            dimension_min_max,
            "\n number_of_days",
            number_of_days,
            "\n start_date",
            start_date,
            "\n end_date",
            end_date,
            "\n total_budget",
            total_budget,
        )

        print(
            "df_optimizer_results_post_min_max \n",
            df_optimizer_results_post_min_max.columns,
        )
        # Table1
        df_table_1_data = df_optimizer_results_post_min_max[
            [
                "dimension",
                "original_median_budget_per_day",
                "recommended_budget_per_day",
                "buget_allocation_old_%",
                "buget_allocation_new_%",
                "recommended_budget_for_n_days",
                'estimated_return_per_day', 
                'estimated_return_%', 
                'estimated_return_for_n_days',
                'current_projections_for_n_days'
            ]
        ]
        print('deeps',df_table_1_data[["buget_allocation_old_%"]])
        # Total values
        df_sum_ = df_table_1_data.sum()
        df_sum_['original_median_budget_per_day'] = df_sum_['original_median_budget_per_day'].round()
        df_sum_['recommended_budget_per_day'] = df_sum_['recommended_budget_per_day'].round()
        df_sum_['buget_allocation_old_%'] = round(df_sum_['buget_allocation_old_%'])
        df_sum_['buget_allocation_new_%'] = round(df_sum_['buget_allocation_new_%'])
        df_sum_['recommended_budget_for_n_days'] = df_sum_['recommended_budget_for_n_days']
        df_sum_['current_projections_for_n_days'] = df_sum_['current_projections_for_n_days']
        df_sum_[df_sum_.index == "dimension"] = "Total"
        df_table_1_data['original_median_budget_per_day'] = df_table_1_data['original_median_budget_per_day'].round()
        df_table_1_data['recommended_budget_per_day'] = df_table_1_data['recommended_budget_per_day'].round()
        df_table_1_data['buget_allocation_old_%'] = df_table_1_data['buget_allocation_old_%']
        df_table_1_data['buget_allocation_new_%'] = df_table_1_data['buget_allocation_new_%']
        df_table_1_data['recommended_budget_for_n_days'] = df_table_1_data['recommended_budget_for_n_days'].round()
        df_table_1_data = df_table_1_data.append(df_sum_, ignore_index=True)
        df_table_1_data = df_table_1_data

        #   To Download CSV
        # df_optimizer_results_post_min_max.to_csv("optimizer_results_post_min_max.csv")
        is_weekly_selected = request.session.get("is_weekly_selected")
        convert_to_weekly_data = request.session.get("convert_to_weekly_data")
        df_table_for_csv = pd.DataFrame()
        if is_weekly_selected or convert_to_weekly_data:
            df_table_for_csv =  df_table_1_data.rename(columns = {'original_median_budget_per_day':'original_median_budget_per_week',
                                           'recommended_budget_per_day':'recommended_budget_per_week',
                                           'recommended_budget_for_n_days':'recommended_budget_for_n_weeks',
                                            'estimated_return_per_day':'estimated_return_per_week',
                                            'estimated_return_for_n_days':'estimated_return_for_n_weeks',
                                            'current_projections_for_n_days':'current_projections_for_n_weeks'
                                           }, inplace = False)
        else:
            df_table_for_csv = df_table_1_data
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
        # json_donut_chart_data = df_donut_chart_data.to_dict()


        print("json_table_1_data", json_table_1_data)
        # print("json_donut_chart_data", json_donut_chart_data)
        print("dict_donut_chart_data", dict_donut_chart_data)
        print("dict_line_chart_data", dict_line_chart_data)
        # Table1
        context["optimizer_download_csv_json"] = json_dumped_optimizer_download_csv
        context["json_table_1_data"] = json_table_1_data
        # context["json_donut_chart_data"] = json_donut_chart_data
        context["dict_donut_chart_data"] = dict_donut_chart_data
        context["dict_line_chart_data"] = dict_line_chart_data
        return JsonResponse(context)
    except Exception as e:
        return JsonResponse({"error": ERROR_DICT[str(e)]}, status=500)
        # messages.error(request, e)
        # return JsonResponse({"error": str(e)}, status=403)

@csrf_exempt
def optimizer_save_the_plan(request):
    try:
        from datetime import date
        today = date.today()
        current_user = request.user
        plan_name = request.GET.get('plan_name')
        count = SavedPlan.objects.count()
        max_plan_id = 0
        if count != 0:
            max_plan_id = SavedPlan.objects.all().order_by("-plan_id")[0].plan_id
        
        plan_from_db = SavedPlan.objects.filter(plan_name=plan_name)
        if plan_from_db:
            return JsonResponse({'message': 'Plan by this name already exists!!!'}, status=400)
        json_table_data = request.GET.get('json_table_data')
        donut_chart_data = request.GET.get('donut_chart_data')
        left_hand_panel_data = request.GET.get('left_hand_panel_data')
        discarded_dimensions_json = request.GET.get('discarded_dimensions_json')
        plan_result_table = json.dumps(json.loads(json_table_data))
        plan_result_donut_chart = json.dumps(json.loads(donut_chart_data))
        left_hand_panel_data = json.dumps(json.loads(left_hand_panel_data))
        discarded_dimensions_json = json.dumps(json.loads(discarded_dimensions_json))
        plan_result_table_path = 'optimizer/'+'plan'+str(max_plan_id+1)+'/plan_result_table.json'
        plan_result_donut_chart_path = 'optimizer/'+'plan'+str(max_plan_id+1)+'/plan_result_donut_chart.json'
        left_hand_panel_data_path = 'optimizer/'+'plan'+str(max_plan_id+1)+'/left_hand_panel_data_path.json'
        discarded_json_path = 'optimizer/'+'plan'+str(max_plan_id+1)+'/discarded_json_path.json'
        json_config_data = ""
        with open('config.json') as config_file:
            json_config_data = json.load(config_file)
        session = boto3.Session(
            aws_access_key_id = json_config_data['ACCESS_KEY_ID'],
            aws_secret_access_key = json_config_data['SECRET_ACCESS_KEY'],
        )
        s3 = session.resource('s3')
        s3object_plan_result_table = s3.Object('optimizer-bkt', plan_result_table_path)
        s3object_plan_result_donut_chart = s3.Object('optimizer-bkt', plan_result_donut_chart_path)
        s3object_left_hand_panel_data_path = s3.Object('optimizer-bkt', left_hand_panel_data_path)
        s3object_discarded_json_path = s3.Object('optimizer-bkt', discarded_json_path)
        s3object_plan_result_table.put(
            Body=(bytes(plan_result_table.encode('UTF-8')))
        )
        s3object_plan_result_donut_chart.put(
            Body=(bytes(plan_result_donut_chart.encode('UTF-8')))
        )
        s3object_left_hand_panel_data_path.put(
            Body=(bytes(left_hand_panel_data.encode('UTF-8')))
        )
        s3object_discarded_json_path.put(
            Body=(bytes(discarded_dimensions_json.encode('UTF-8')))
        )
        s = SavedPlan.objects.create(plan_name=plan_name,
                                    plan_date=today,
                                    user=current_user,
                                    plan_result_table_path=plan_result_table_path,
                                    plan_result_donut_chart_path=plan_result_donut_chart_path)
        s.save()
        return JsonResponse({"message": "plan saved successfully"}, status=200)
    except Exception as e:
        return HttpResponse(ERROR_DICT[str(e)], status=500)
