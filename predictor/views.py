from datetime import datetime
import os
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.shortcuts import render
import numpy as np
import pandas as pd
from urllib3 import HTTPResponse
from .predictor import predictor as predictor_ml, predictor_with_seasonality
from .predictor import (
    predict_dimesion,
    predict_dimesion_with_seasonality,
    progress_bar_var,
)
from .save_pdf import plot_curve
import json
# from django.template.loader import render_to_string
import functools
from django.http import FileResponse

ERROR_DICT = {
    "5002": "Value Error",
    "5003": "Type Error",
    "5004": "Incorrect Date Format",
}
# for production environment
# UPLOAD_FOLDER = "/var/www/optimizer/data/"
# PREDICTOR_UPLOAD_FOLDER = "/var/www/optimizer/Predictor_pdf/"
# for local environment
UPLOAD_FOLDER = "data/"
PREDICTOR_UPLOAD_FOLDER = "Predictor_pdf/"
TEMP_ERROR_DICT = {"4002": "Value Error"}

# Global variables
global_df_param = None
global_df_score_final = None
global_scatter_plot_df = None
global_drop_dimension = None
global_d_cpm = None
global_weekly_predictions_df = None
global_monthly_predictions_df = None
# Create your views here.

# helper function
# save the latest file as DF (pickle file) from predictor page ( on submit and on date change )
def save_predictor_page_latest_data(df_predictor_page_latest_data, _uuid):
    scatter_plot_df = global_scatter_plot_df
    df_predictor_page_latest_data.to_pickle(
        UPLOAD_FOLDER+"df_predictor_page_latest_data_{}.pkl".format(_uuid)
    )
    scatter_plot_df.to_pickle(
        UPLOAD_FOLDER + "scatter_plot_df_{}.pkl".format(_uuid)
    )


@login_required
def predictor_home_page(request):
    try:
        context = {}
        file_exists = os.path.exists(
            UPLOAD_FOLDER + "{}_agg_data.pkl".format(request.session.get("_uuid"))
        )
        is_weekly_selected = request.session.get("is_weekly_selected")
        convert_to_weekly_data = request.session.get("convert_to_weekly_data")
        is_predict_page_submit_success = request.session.get(
            "is_predict_page_submit_success"
        )
        cpm_checked = request.session.get("cpm_checked")
        if file_exists and is_predict_page_submit_success:
            context["run_wondow_onload"] = 1
            agg_data = pd.read_pickle(
                UPLOAD_FOLDER + "{}_agg_data.pkl".format(request.session.get("_uuid"))
            )
            agg_data = pd.read_pickle(
                UPLOAD_FOLDER + "{}_agg_data.pkl".format(request.session.get("_uuid"))
            )
            unique_dimensions = agg_data["dimension"].unique()
            context['unique_dimensions'] = unique_dimensions
        elif file_exists and is_predict_page_submit_success == 0:
            agg_data = pd.read_pickle(
                UPLOAD_FOLDER + "{}_agg_data.pkl".format(request.session.get("_uuid"))
            )
            unique_dimensions = agg_data["dimension"].unique()
            context['unique_dimensions'] = unique_dimensions
        else:
            context["run_wondow_onload"] = 0

        if is_weekly_selected:
            print(f"is_weekly_selected : {is_weekly_selected}")
            context["is_weekly_selected"] = int(is_weekly_selected)
        if convert_to_weekly_data:
            context["convert_to_weekly_data"] = int(convert_to_weekly_data) 

        if cpm_checked == "True":
            print("returning cpm get request")
            context["cpm_message"] = "cpm selected"
        return render(request, "predictor/predictor.html", context)

    except Exception as exp:
        print(f"Exception in predictor_home_page: {exp}")
        return JsonResponse({"error": TEMP_ERROR_DICT[str(exp)]}, status=403)


def get_predictor_start_and_end_dates(request):
    try:
        context = {}
        unique_dimensions_json = {}
        file_exists = os.path.exists(
            UPLOAD_FOLDER + "{}_agg_data.pkl".format(request.session.get("_uuid"))
        )
        if file_exists:
            agg_data = pd.read_pickle(
                UPLOAD_FOLDER + "{}_agg_data.pkl".format(request.session.get("_uuid"))
            )
            unique_dimensions = agg_data["dimension"].unique()
            if request.session.get("predictor_unique_dimensions_json") is not None:
                unique_dimensions_keys_from_previous_data = list(request.session["predictor_unique_dimensions_json"].keys())
            # both the dimension sets should be exactly same to enter this condition 
            # because we are exacty replicating same data
            if request.session.get("predictor_unique_dimensions_json") is not None and functools.reduce(lambda x, y : x and y, map(lambda p, q: p == q,unique_dimensions,unique_dimensions_keys_from_previous_data), True):
                predictor_unique_dimensions_json = request.session["predictor_unique_dimensions_json"]
                for dimension in unique_dimensions:
                    # get Start and End dates
                    start_date = predictor_unique_dimensions_json[dimension]['0']
                    end_date = predictor_unique_dimensions_json[dimension]['1'] 
                    unique_dimensions_json[dimension] = [start_date, end_date]
            # user will enter here either is a new to predictor or has chaged dimensions on explorer 
            else:
                for dimension in unique_dimensions:
                    agg_data["date"] = pd.to_datetime(agg_data["date"])
                    # get Start and End dates
                    start_date = agg_data[agg_data["dimension"] == dimension]["date"].min().strftime("%m/%d/%y")
                    end_date = agg_data[agg_data["dimension"] == dimension]["date"].max().strftime("%m/%d/%y")
                    unique_dimensions_json[dimension] = [start_date, end_date]
        else:
            start_date = "01/25/22"
            end_date = "02/15/22"
        context['unique_dimensions_json'] = unique_dimensions_json

        return JsonResponse(context, status=200)

    except Exception as exp:
        print(f"Exception in get_predictor_start_and_end_dates: {exp}")
        return JsonResponse(
            {"error": f"Error in get_predictor_start_and_end_dates : {exp}"}, status=403
        )

def create_pdf_for_multiple_plots(multi_line_chart_json, seasonality, cpm_checked):
    multi_line_chart_data = {}
    old_key = ""
    for obj in multi_line_chart_json:
        dimension_key = obj['dimension']
        if old_key != dimension_key:
            multi_line_chart_data[dimension_key] = {}
            multi_line_chart_data[dimension_key]['spend'] = []
            if seasonality == 1:
                 multi_line_chart_data[dimension_key]['spend_prediction'] = []
            if cpm_checked == "True":
                multi_line_chart_data[dimension_key]['impression'] = []
            multi_line_chart_data[dimension_key]['target'] = []
            multi_line_chart_data[dimension_key]['predictions'] = []
            old_key = dimension_key
        multi_line_chart_data[dimension_key]['spend'].append(obj['spend'])
        multi_line_chart_data[dimension_key]['target'].append(obj['target'])
        multi_line_chart_data[dimension_key]['predictions'].append(obj['predictions'])
        if seasonality == 1:
            multi_line_chart_data[dimension_key]['spend_prediction'].append(obj['spend_prediction'])
        if cpm_checked == "True":
             multi_line_chart_data[dimension_key]['impression'].append(obj['impression'])
    return multi_line_chart_data


def get_multi_line_chart_data2(multi_line_chart_json, cpm_checked):
    multi_line_chart_data2 = []
    old_key = ""
    multi_line_chart_obj = {}
    predictions_spend_obj = {}
    for index, obj in enumerate(multi_line_chart_json, start=0):
        dimension_key = obj['dimension']
        if old_key != dimension_key:
            if bool(multi_line_chart_obj) is True:
                multi_line_chart_data2.append(multi_line_chart_obj)
            multi_line_chart_obj = {}
            multi_line_chart_obj["name"] = dimension_key
            multi_line_chart_obj["values"] = []
        predictions_spend_obj = {}
        if cpm_checked == "True":
            predictions_spend_obj["impression"] = obj["impression"]
            predictions_spend_obj["predictions"] = obj["predictions"]
            multi_line_chart_obj["values"].append(predictions_spend_obj)
        else:
            predictions_spend_obj["spend"] = obj["spend"]
            predictions_spend_obj["predictions"] = obj["predictions"]
            multi_line_chart_obj["values"].append(predictions_spend_obj)
        if index == len(multi_line_chart_json)-1:
            multi_line_chart_data2.append(multi_line_chart_obj)
        old_key = dimension_key
    return multi_line_chart_data2


def predictor_ajax_left_panel_submit(request):
    print("predictor_ajax_left_panel_submit")
    try:
        context = {}
        default_dim = "0"
        body = json.loads(request.body)
        seasonality = int(body['seasonality'])
        cpm_checked = request.session.get("cpm_checked")
        request.session["seasonality"] = seasonality
        print("seasonality", seasonality)

        # print(request.session.get('cpm_checked'), "cpm on submit")
        # Adi Func
        agg_data = pd.read_pickle(
            UPLOAD_FOLDER + "{}_agg_data.pkl".format(request.session.get("_uuid"))
        )

        if seasonality == 1:
            print("Running seasonality predictor_with_seasonality")
            object_predictor_ml = predictor_with_seasonality(agg_data)
        else:
            print("Running seasonality predictor_ml")
            predictor_unique_dimensions_json = json.loads(body["stringified_unique_dimensions_json"])
            request.session["predictor_unique_dimensions_json"] = pd.DataFrame(predictor_unique_dimensions_json).to_dict()
            for key in predictor_unique_dimensions_json:
                predictor_unique_dimensions_json[key][0] = datetime.strptime(predictor_unique_dimensions_json[key][0] , "%m/%d/%y").date()
                predictor_unique_dimensions_json[key][1] = datetime.strptime(predictor_unique_dimensions_json[key][1] , "%m/%d/%y").date()
            object_predictor_ml = predictor_ml(agg_data, predictor_unique_dimensions_json)

        if cpm_checked == "True":
            print("reading impdata")
            # object_predictor_ml.predict_dimesion("aaaa",, "budget", "True")
            # data = pd.read_csv("data/impression_sample.csv")
            context["cpm_message"] = "cpm selected"
            try:
                sort = "impression"
                (
                    df_param,
                    df_score_final,
                    scatter_plot_df,
                    drop_dimension,
                    d_cpm,
                    df_spend_dis
                ) = object_predictor_ml.execute()
            except Exception as error:
                print(str(error))
                return JsonResponse({"error": str(error)}, status=500)


        else:
            print("spend data")
            # data = pd.read_csv("data/scatterplot_data.csv")
            sort = "spend"
            try:
                (
                    df_param,
                    df_score_final,
                    scatter_plot_df,
                    drop_dimension,
                    df_spend_dis
                ) = object_predictor_ml.execute()
            except Exception as error:
                print(str(error))
                return JsonResponse({"error": str(error)}, status=500)                
            d_cpm = None
        weekly_predictions_df = pd.DataFrame()
        monthly_predictions_df = pd.DataFrame()
        if seasonality == 1:
            weekly_predictions_df = scatter_plot_df[['dimension', 'weekday_', 'weekly_prediction']]
            monthly_predictions_df = scatter_plot_df[['dimension', 'month_', 'monthly_prediction']]
        request.session['df_spend_dis'] = df_spend_dis.to_dict()
        global global_df_param
        global global_df_score_final
        global global_scatter_plot_df
        global global_drop_dimension
        global global_d_cpm
        global global_weekly_predictions_df
        global global_monthly_predictions_df
        global_df_param = df_param
        global_df_score_final = df_score_final
        global_scatter_plot_df = scatter_plot_df
        global_drop_dimension = drop_dimension
        global_d_cpm = d_cpm
        multi_line_chart_df = scatter_plot_df
        global_weekly_predictions_df = weekly_predictions_df
        global_monthly_predictions_df = monthly_predictions_df
        if cpm_checked == "True":
            sort_multi = ['dimension', 'impression']
            max_spend = multi_line_chart_df.loc[multi_line_chart_df["impression"].idxmax()]["impression"]
        else:
            sort_multi = ['dimension', 'spend']
            max_spend = multi_line_chart_df.loc[multi_line_chart_df["spend"].idxmax()]["spend"]
        multi_line_chart_df = multi_line_chart_df.sort_values(by=sort_multi)
        max_predictions = multi_line_chart_df.loc[multi_line_chart_df["predictions"].idxmax()]["predictions"]
        # multi_line_chart_df = multi_line_chart_df[(multi_line_chart_df["dimension"] == "SEM Brand")]
        multi_line_chart_json = multi_line_chart_df.to_dict("records")
        multi_line_chart_data2 = get_multi_line_chart_data2(multi_line_chart_json, cpm_checked)
        # weekly_predictions_dataframes
        _uuid = request.session.get("_uuid")
        save_predictor_page_latest_data(df_param, _uuid)
        df_score_final = df_score_final[
            df_score_final["dimension"].isin(scatter_plot_df["dimension"].unique())
        ]
        df_score_final.sort_values(
            by=["data_points_post_outlier_treatment"], ascending=False, inplace=True
        )
        # unique_dim = [
        #     col
        #     for col in np.array(df_score_final["dimension"])
        #     if (col in np.array(scatter_plot_df["dimension"].unique()))
        # ]
        unique_dim = list(df_score_final["dimension"])
        print("uniq_dim", unique_dim)
        default_dim = unique_dim[0]
        request.session["predictor_default_dim"] = default_dim
        # dimension_value_selector = list(set(unique_dim) - set([default_dim]))
        unique_dim.remove(default_dim)
        dimension_value_selector = unique_dim
        if cpm_checked == "True":
            # as cpm is selected
            enter_cpm = d_cpm[default_dim]
            enter_cpm = round(enter_cpm, 2)
            print("enter_cpm", enter_cpm)
            context["enter_cpm"] = enter_cpm

        # print("default_dim", default_dim)
        # print("df_param\n", df_param)
        print(
            "df_score_final\n",
            df_score_final[["dimension", "data_points_post_outlier_treatment"]],
        )
        # print("scatter_plot_df\n", scatter_plot_df)
        # print("drop_dimension\n", drop_dimension)
        print("dimension_value_selector\n", dimension_value_selector)
        # scatter_plot_df = scatter_plot_df.sort()
        scatter_plot_df = scatter_plot_df.sort_values(by=[sort])
        # impression_data = scatter_plot_df.sort_values(by=['spend'])

        # default_dim = list(scatter_plot_df["dimension"].unique())[0]

        print("***********************default dimension**************")

        # scatter_plot_df = scatter_plot_df[(scatter_plot_df["dimension"]==default_dim) & (scatter_plot_df["date"] >= start_date) & (scatter_plot_df["date"] <= end_date)]
        # scatter_plot_df['date'] = pd.to_datetime(scatter_plot_df['date']).dt.strftime('%m-%d-%y')
        print("date is working for predictor")
        scatter_plot_df["date"] = pd.to_datetime(scatter_plot_df["date"]).dt.date

        if seasonality == 1:
            scatter_plot_df = scatter_plot_df[
                (scatter_plot_df["dimension"] == default_dim)
            ]

        else:
            scatter_plot_df = scatter_plot_df[
                (scatter_plot_df["dimension"] == default_dim)
               ]
        multi_line_chart_data = create_pdf_for_multiple_plots(multi_line_chart_json, seasonality, cpm_checked)
        plot_curve(multi_line_chart_data, seasonality, cpm_checked, df_score_final, request)
        df_score_final = df_score_final[(df_score_final["dimension"] == default_dim)]
        if seasonality == 1:
            weekly_predictions_df = weekly_predictions_df[
                            (weekly_predictions_df["dimension"] == default_dim)
                                ].drop_duplicates()
            monthly_predictions_df = monthly_predictions_df[
                            (monthly_predictions_df["dimension"] == default_dim)
                                ].drop_duplicates()
        monthly_predictions_json = monthly_predictions_df.to_dict("records")
        weekly_predictions_json = weekly_predictions_df.to_dict("records")
        predictor_table_data = df_score_final.to_dict("records")
        # cpm_value = d_cpm[default_dim]
        # print(cpm_value)
        print("scatter_plot_df\n", scatter_plot_df)
        print("df_score_final_dict\n", predictor_table_data)
        print("***********************dimension value selector**************")
        scatter_plot_df = scatter_plot_df.to_dict("records")
        predictor_table_data = df_score_final.to_dict("records")

        # impression_data = impression_data.to_dict('records')
        print("*******************************")
        # print(scatter_plot_df)
        context["dimension_value_selector"] = dimension_value_selector
        context["scatterplot_data"] = scatter_plot_df
        # context["impression_scatterplot_data"] = impression_data
        context["unique_dimension"] = unique_dim
        context["default_dimension"] = default_dim
        context["predictor_table_data"] = predictor_table_data
        if seasonality == 1:
            context["weekly_predictions_json"] = weekly_predictions_json
            context["monthly_predictions_json"] = monthly_predictions_json
        context["drop_dimension"] = drop_dimension
        context["seasonality"] = seasonality
        context["multi_line_chart_data2"] = multi_line_chart_data2
        context["max_spend"] = max_spend
        context["max_predictions"] = max_predictions
        request.session["is_predict_page_submit_success"] = 1
        request.session["drop_dimension"] = drop_dimension
        return JsonResponse(context)
    except Exception as exp:
        print(f"\nException in predictor_ajax_left_panel_submit{exp}")
        return JsonResponse({"error": TEMP_ERROR_DICT[str(exp)]}, status=403)


def predictor_ajax_date_dimension_onchange(request):
    print("predictor_ajax_date_dimension_onchange")
    try:
        context = {}
        seasonality = int(request.session.get("seasonality"))
        default_dim = request.GET.get("dimension_value_selector")
        request.session["predictor_default_dim"] = default_dim
        print("default dim on chnage", seasonality)
        print("default dim on chnage", default_dim)

        if request.session.get("cpm_checked") == "True":
            print("CPM selected ,reading impdata")
            # data = pd.read_csv("data/impression_sample.csv")
            context["cpm_message"] = "cpm selected"
            sort = "impression"
            df_param = global_df_param
            df_score_final = global_df_score_final
            scatter_plot_df = global_scatter_plot_df
            drop_dimension = global_drop_dimension
            weekly_predictions_df = global_weekly_predictions_df
            monthly_predictions_df = global_monthly_predictions_df
            d_cpm = global_d_cpm
            # as cpm is selected
            enter_cpm = d_cpm[default_dim]
            enter_cpm = round(enter_cpm, 2)
            print("enter_cpm", enter_cpm)
            context["enter_cpm"] = enter_cpm

        else:
            print("CPM NOT selected , spend data")
            # data = pd.read_csv("data/scatterplot_data.csv")
            sort = "spend"
            df_param = global_df_param
            df_score_final = global_df_score_final
            scatter_plot_df = global_scatter_plot_df
            drop_dimension = global_drop_dimension
            d_cpm = None
            weekly_predictions_df = global_weekly_predictions_df
            monthly_predictions_df = global_monthly_predictions_df
        print("df_param\n", df_param)
        print("df_score_final\n", df_score_final)
        print("scatter_plot_df\n", scatter_plot_df)
        print("drop_dimension\n", drop_dimension)

        df_score_final.sort_values(
            by=["data_points_post_outlier_treatment"], ascending=False, inplace=True
        )
        # converting from string to datetime
        scatter_plot_df["date"] = pd.to_datetime(scatter_plot_df["date"]).dt.date

        if seasonality:
            scatter_plot_df = scatter_plot_df[
                (scatter_plot_df["dimension"] == default_dim)
            ]
        else:
            # str_start_date = request.GET.get("start_date")
            # str_end_date = request.GET.get("end_date")
            # request.session["predictor_start_date"] = str_start_date
            # request.session["predictor_end_date"] = str_end_date
            # start_date = datetime.strptime(str_start_date, "%m-%d-%y").date()
            # end_date = datetime.strptime(str_end_date, "%m-%d-%y").date()
            # print("start_date, end_date", start_date, end_date)
            scatter_plot_df = scatter_plot_df[
                (scatter_plot_df["dimension"] == default_dim)
            ]
        if seasonality == 1:
            weekly_predictions_df = weekly_predictions_df[
                        (weekly_predictions_df["dimension"] == default_dim)
                            ].drop_duplicates()
            monthly_predictions_df = monthly_predictions_df[
                        (monthly_predictions_df["dimension"] == default_dim)
                            ].drop_duplicates()        
        weekly_predictions_json = weekly_predictions_df.to_dict("records")
        monthly_predictions_json = monthly_predictions_df.to_dict("records")
        scatter_plot_df = scatter_plot_df.sort_values(by=[sort])
        scatter_plot_df = scatter_plot_df.to_dict("records")
        df_score_final = df_score_final[(df_score_final["dimension"] == default_dim)]
        predictor_table_data = df_score_final.to_dict("records")
        context["scatterplot_data"] = scatter_plot_df
        context["predictor_table_data"] = predictor_table_data
        context["drop_dimension"] = drop_dimension
        context["weekly_predictions_json"] = weekly_predictions_json
        context["monthly_predictions_json"] = monthly_predictions_json
        return JsonResponse(context, status=200)

    except Exception as exp:
        return JsonResponse({"error": TEMP_ERROR_DICT[str(exp)]}, status=403)


def get_is_weekly_selected(request):
    try:
        context = {}
        print("get_is_weekly_selected")
        is_weekly_selected = int(request.session.get("is_weekly_selected"))
        print(
            "\n is_weekly_Selected",
            is_weekly_selected,
        )
        if is_weekly_selected:
            context["is_weekly_selected"] = is_weekly_selected

        else:
            context["is_weekly_selected"] = 0

        return JsonResponse(context)
    except Exception as exp:
        message = f"Exception in get_weekly_selected {exp}"
        print(message)
        return JsonResponse({"error": message}, status=403)


def get_convert_to_weekly_data(request):
    try:
        context = {}
        print("get_convert_to_weekly_data")
        convert_to_weekly_data = int(request.session.get("convert_to_weekly_data"))
        print(
            "\n convert_to_weekly_data",
            convert_to_weekly_data
        )
        if convert_to_weekly_data:
            context["convert_to_weekly_data"] = convert_to_weekly_data

        else:
            context["convert_to_weekly_data"] = 0

        return JsonResponse(context)
    except Exception as exp:
        message = f"Exception in get_weekly_selected {exp}"
        print(message)
        return JsonResponse({"error": message}, status=403)


def get_seasonality_from_session(request):
    try:
        context = {}
        print("get_seasonality_from_session")
        seasonality_from_session = int(request.session.get("seasonality"))
        print(
            "\n seasonality_from_session",
            seasonality_from_session,
        )
        if seasonality_from_session:
            context["seasonality_from_session"] = seasonality_from_session

        else:
            context["seasonality_from_session"] = 0

        return JsonResponse(context)
    except Exception as exp:
        message = f"Exception in get_seasonality_from_session {exp}"
        print(message)
        return JsonResponse({"error": message}, status=403)


def predictor_ajax_predictor_send_value(request):
    try:
        print("predictor_ajax_predictor_send_value")
        # dimension = "Audio"
        df_predictor_page_latest_data = pd.read_pickle(
            UPLOAD_FOLDER + "df_predictor_page_latest_data_{}.pkl".format(
                request.session.get("_uuid")
            )
        )

        # variables
        context = {}
        dimension = request.GET.get("dimension")
        seasonality = int(request.session.get("seasonality"))

        budget = int(request.GET.get("budget"))
        predicted_value = 0
        start_date = None
        end_date = None
        cpm_value = None
        total_days = None
        if seasonality:
            start_date = datetime.strptime(
                request.GET.get("start_date"), "%m-%d-%y"
            ).date()
            end_date = datetime.strptime(request.GET.get("end_date"), "%m-%d-%y").date()
        else:
            total_days = int(request.GET.get("total_days"))
        if request.GET.get("cpm_value"):
            cpm_value = float(request.GET.get("cpm_value"))
        else:
            cpm_value = None

        print(
            "\n seasonality",
            seasonality,
            "\n dimension",
            dimension,
            "\n start_date",
            start_date,
            "\n end_date",
            end_date,
            "\n total_days",
            total_days,
            "\n budget",
            budget,
            "\n cpm_value",
            cpm_value,
        )

        # if ("GET" == request.method) & (request.session.get("cpm_checked") == "True"):
        #     print(
        #         "predictor_ajax_predictor_send_value : CPM selected value: ", cpm_value
        #     )
        #     predicted_value = predict_dimesion(
        #         df_predictor_page_latest_data,
        #         dimension,
        #         [start_date, end_date],
        #         budget,
        #         cpm_value,
        #     )
        # else:
        #     print("predictor_ajax_predictor_send_value : cpm NOT selected")
        #     predicted_value = predict_dimesion(
        #         df_predictor_page_latest_data, dimension, [start_date, end_date], budget
        #     )
        if seasonality:
            predicted_value = predict_dimesion_with_seasonality(
                df_predictor_page_latest_data,
                dimension,
                [start_date, end_date],
                budget,
                cpm=cpm_value,
            )

        else:
            predicted_value = predict_dimesion(
                df_predictor_page_latest_data,
                dimension,
                total_days,
                budget,
                cpm=cpm_value,
            )

        print("predicted_value sent to forntend", predicted_value)
        context["predicted_value"] = round(predicted_value, 2)
        return JsonResponse(context)
    except Exception as exp:
        message = f"Exception in predictor_ajax_predictor_send_value {exp}"
        print(message)
        return JsonResponse({"error": message}, status=403)


def predictor_ajax_predictor_discard(request):
    try:
        print("predictor_ajax_predictor_discard")

        # variables
        context = {}
        discarded_items = request.GET.get("discarded_items")
        drop_dimension_from_session = request.session.get("drop_dimension")
        request.session["discarded_items"] = discarded_items
        discarded_items_from_session = request.session["discarded_items"]
        print(discarded_items_from_session)

        # testing
        # reading the df
        discarded_items_from_session_array = discarded_items_from_session.split(",")
        # adding discarded dimensions by user to drop dimensions
        for item in discarded_items_from_session_array:
            drop_dimension_from_session.append(item)
            
        df_predictor_page_latest_data = pd.read_pickle(
            UPLOAD_FOLDER + "df_predictor_page_latest_data_{}.pkl".format(
                request.session.get("_uuid")
            )
        )
        # updating the df by removing the distacarded rows
        df_predictor_page_latest_data = df_predictor_page_latest_data[
            ~df_predictor_page_latest_data["dimension"].isin(
                discarded_items_from_session_array
            )
        ]

        # saving( overwriting) the latest df
        _uuid = request.session.get("_uuid")
        save_predictor_page_latest_data(df_predictor_page_latest_data, _uuid)
        # testing

        context["discarded_items"] = discarded_items_from_session_array
        return JsonResponse(context)
    except Exception as exp:
        print(exp)
        return JsonResponse({"error": TEMP_ERROR_DICT[str(exp)]}, status=403)


def get_progress_bar_dict(request):
    progress_var = progress_bar_var()
    return JsonResponse({"prog_variables": progress_var}, status=200)


def predictor_delete(request):
    context = {}
    type = request.GET.get("type")
    if type == "delete":
        budget = request.GET.get("budget")
        start_date = request.GET.get("start_date")
        end_date = request.GET.get("end_date")
        print(budget)
        print(start_date)
        print(end_date)
        context["message"] = "Successfully deleted"
        return JsonResponse(context)

    return render(request, "predictor/predictor.html", context)


@login_required
def predictor_window_on_load(request):
    print("predictor_window_on_load")
    try:
        context = {}
        seasonality = int(request.session.get("seasonality"))
        default_dim = request.session.get("predictor_default_dim")
        cpm_checked = request.session.get("cpm_checked")
        predictor_unique_dimensions_json = request.session["predictor_unique_dimensions_json"]
        print("seasonality", seasonality)
        print("default_dim", default_dim)
        print("cpm_checked", cpm_checked)

        if cpm_checked == "True":
            print("CPM selected ,reading impdata")
            # data = pd.read_csv("data/impression_sample.csv")
            context["cpm_message"] = "cpm selected"
            sort = "impression"
            df_param = global_df_param
            df_score_final = global_df_score_final
            scatter_plot_df = global_scatter_plot_df
            drop_dimension = global_drop_dimension
            d_cpm = global_d_cpm
            weekly_predictions_df = global_weekly_predictions_df
            monthly_predictions_df = global_monthly_predictions_df
            # as cpm is selected
            enter_cpm = d_cpm[default_dim]
            enter_cpm = round(enter_cpm, 2)
            print("enter_cpm", enter_cpm)
            context["enter_cpm"] = enter_cpm

        else:
            print("CPM NOT selected , spend data")
            # data = pd.read_csv("data/scatterplot_data.csv")
            sort = "spend"
            df_param = global_df_param
            df_score_final = global_df_score_final
            scatter_plot_df = global_scatter_plot_df
            drop_dimension = global_drop_dimension
            weekly_predictions_df = global_weekly_predictions_df
            monthly_predictions_df = global_monthly_predictions_df
            d_cpm = None
        multi_line_chart_df = scatter_plot_df
        if cpm_checked == "True":
            sort_multi = ['dimension', 'impression']
            max_spend = multi_line_chart_df.loc[multi_line_chart_df["impression"].idxmax()]["impression"]
        else:
            sort_multi = ['dimension', 'spend']
            max_spend = multi_line_chart_df.loc[multi_line_chart_df["spend"].idxmax()]["spend"]
        multi_line_chart_df = multi_line_chart_df.sort_values(by=sort_multi)
        
        max_predictions = multi_line_chart_df.loc[multi_line_chart_df["predictions"].idxmax()]["predictions"]
        # multi_line_chart_df = multi_line_chart_df[(multi_line_chart_df["dimension"] == "SEM Brand")]
        multi_line_chart_json = multi_line_chart_df.to_dict("records")
        multi_line_chart_data2 = get_multi_line_chart_data2(multi_line_chart_json, cpm_checked)
        df_score_final = df_score_final[
            df_score_final["dimension"].isin(scatter_plot_df["dimension"].unique())
        ]
        df_score_final.sort_values(
            by=["data_points_post_outlier_treatment"], ascending=False, inplace=True
        )
        # unique_dim = [
        #     col
        #     for col in np.array(df_score_final["dimension"])
        #     if (col in np.array(scatter_plot_df["dimension"].unique()))
        # ]
        unique_dim = list(df_score_final["dimension"])
        default_dim = unique_dim[0]
        unique_dim.remove(default_dim)
        dimension_value_selector = unique_dim
        # converting from string to datetime
        scatter_plot_df["date"] = pd.to_datetime(scatter_plot_df["date"]).dt.date

        print("df_param\n", df_param)
        print("df_score_final\n", df_score_final)
        print("scatter_plot_df\n", scatter_plot_df)
        print("unique_dim\n", unique_dim)
        print("default_dim\n", default_dim)
        print("dimension_value_selector\n", dimension_value_selector)
        if seasonality == 1:
            weekly_predictions_df = weekly_predictions_df[
                            (weekly_predictions_df["dimension"] == default_dim)
                                ].drop_duplicates()
            monthly_predictions_df = monthly_predictions_df[
                            (monthly_predictions_df["dimension"] == default_dim)
                                ].drop_duplicates()
        weekly_predictions_json = weekly_predictions_df.to_dict("records")
        monthly_predictions_json = monthly_predictions_df.to_dict("records")
        if seasonality:
            scatter_plot_df = scatter_plot_df[
                (scatter_plot_df["dimension"] == default_dim)
            ]

        else:
            start_date = datetime.strptime(predictor_unique_dimensions_json[default_dim]['0'] , "%m/%d/%y").date()
            end_date = datetime.strptime(predictor_unique_dimensions_json[default_dim]['1'] , "%m/%d/%y").date()
            scatter_plot_df = scatter_plot_df[
                    (scatter_plot_df["dimension"] == default_dim)
                    & (scatter_plot_df["date"] >= start_date)
                    & (scatter_plot_df["date"] <= end_date)
                ]

        scatter_plot_df = scatter_plot_df.sort_values(by=[sort])
        scatter_plot_df = scatter_plot_df.to_dict("records")
        df_score_final = df_score_final[(df_score_final["dimension"] == default_dim)]
        predictor_table_data = df_score_final.to_dict("records")

        context["scatterplot_data"] = scatter_plot_df
        context["predictor_table_data"] = predictor_table_data
        context["drop_dimension"] = drop_dimension
        context["dimension_value_selector"] = dimension_value_selector
        context["unique_dimension"] = unique_dim
        context["default_dimension"] = default_dim
        context["seasonality"] = seasonality
        if seasonality == 1:
            context["weekly_predictions_json"] = weekly_predictions_json
            context["monthly_predictions_json"] = monthly_predictions_json
        context["multi_line_chart_data2"] = multi_line_chart_data2
        context["max_spend"] = max_spend
        context["max_predictions"] = max_predictions
        return JsonResponse(context, status=200)

    except Exception as exp:
        return JsonResponse({"error": TEMP_ERROR_DICT[str(exp)]}, status=403)

def download_predictor_curves_pdf(request):
    try:
        context = {}
         # Get the file path of the PDF file
        pdf_file = PREDICTOR_UPLOAD_FOLDER+"predictor_" + request.session.get("_uuid") + ".pdf"  
        pdf = open(pdf_file, 'rb')
        response = FileResponse(pdf, content_type='application/pdf')
        response['Content-Disposition'] = 'attachment; filename="predictor.pdf"'
        return response       
    except Exception as e:
        print("Error download_sample_csv", e)
        return JsonResponse({"error": TEMP_ERROR_DICT[str(e)]}, status=403)
