from django.contrib.auth.decorators import login_required
import json
import pickle
from django.http import HttpResponse, JsonResponse
from django.shortcuts import render, redirect
import numpy as np
import pandas as pd
from django.core.files.storage import FileSystemStorage
from .explorer import explorer
import os
from django.conf import settings
from datetime import datetime
import uuid
import chardet
# from django.template.loader import render_to_string
# from predictor import predictor as predictor_ml


ERROR_DICT = {
    "5002": "Value Error",
    "5003": "Type Error",
    "5004": "Incorrect Date Format",
}
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

# global_df_param = NULL
# global_df_score_final = NULL
# global_scatter_plot_df = NULL
# global_drop_dimension = NULL
# d_cpm = NULL


@login_required
def explorer_home_page(request):
    context1 = {}
    print("Home")

    if "GET" == request.method:
        print("returning get request")

        # Persitance View
        if request.session.get("_uuid", False):
            context = {}
            print("_uuid", request.session["_uuid"])
            data = pd.read_pickle(UPLOAD_FOLDER+"{}.pkl".format(request.session.get("_uuid")))
            selectors_data_list = list(data.columns)
            context["selector"] = selectors_data_list
            file_exists = os.path.exists(
                "data/{}_agg_data.pkl".format(request.session.get("_uuid"))
            )
            context["run_wondow_onload"] = 1 if file_exists else 0

            return render(request, "home/explorer.html", context)
        else:
            return render(request, "home/explorer.html", {})
        # Persitance View
    else:
        if "Upload" in request.POST:
            try:
                context = {}
                _uuid = uuid.uuid4()
                excel_file = request.FILES["excel_file"]
                fs = FileSystemStorage(
                    location=UPLOAD_FOLDER
                )  # defaults to   MEDIA_ROOT
                if os.path.exists(UPLOAD_FOLDER + excel_file.name):
                    os.remove(UPLOAD_FOLDER + excel_file.name)
                fs.save(excel_file.name, excel_file)
                print("saved successfuly")
                encoding_format = ""
                with open(UPLOAD_FOLDER + excel_file.name, 'rb') as rawdata:
                    result = chardet.detect(rawdata.read(100000))
                    encoding_format = result['encoding']
                if encoding_format == 'MacRoman':
                    data = pd.read_csv(UPLOAD_FOLDER + excel_file.name, encoding="MacRoman")
                else:
                    data = pd.read_csv(UPLOAD_FOLDER + excel_file.name)
                # request.session['data'] = data.to_dict('dict')
                data.to_pickle(UPLOAD_FOLDER+"{}.pkl".format(_uuid))
                request.session["_uuid"] = str(_uuid)
                # request.session['file_name'] = excel_file.name

                print(data.columns)
                selectors_data_list = list(data.columns)  # column header for dropdown
                print("list created")
                request.session["is_weekly_selected"] = 0
                request.session["cpm_checked"] = "False"

                context["selector"] = selectors_data_list

                return render(request, "home/explorer.html", context)
            except Exception as e:
                print(f"Error uploading the file : {e}")
                return JsonResponse({"error": "Error uploading the file"}, status=403)

    # return render(request, 'home/explorer.html', context1)
    return JsonResponse(context1)


# def login(request):
#     return render(request, "optimizer/login.html")


def download_sample_csv(request):
    print("download_sample_csv")
    try:
        context = {}
        df_sample_download_file = pd.read_csv(UPLOAD_FOLDER +"sample_download_file.csv")
        csv_sample_download_file = df_sample_download_file.to_csv(index=False)
        json_dumped_sample_download_file = json.dumps(
            csv_sample_download_file, separators=(",", ":")
        )
        # print("json_dumped_sample_download_file", json_dumped_sample_download_file)
        context["json_dumped_sample_download_file"] = json_dumped_sample_download_file
        return JsonResponse(context)
    except Exception as e:
        print("Error download_sample_csv", e)
        return HttpResponse("Error Sending sample file to download", status=403)

def convert_to_weekly_data(request):
    try:
        print("convert_to_weekly_data")
        request.session["convert_to_weekly_data"] = request.GET.get("convert_to_weekly_data")
        convert_to_weekly_data_from_session = request.session.get("convert_to_weekly_data")
        return JsonResponse(
            {
                "message": f"Updated convert_to_weekly_data = {convert_to_weekly_data_from_session}"
            },
            status=200,
        )

    except Exception as e:
        print(f"Exception in convert_to_weekly_data :\n {e}")
        return JsonResponse(
            {
                "error": f"Exception in convert_to_weekly_data : {convert_to_weekly_data_from_session}"
            },
            status=403,
        )


def is_weekly_selected(request):
    try:
        print("is_weekly_selected")
        request.session["is_weekly_selected"] = request.GET.get("is_weekly_selected")
        is_weekly_selected_from_session = request.session.get("is_weekly_selected")
        return JsonResponse(
            {
                "message": f"Updated is_weekly_selected = {is_weekly_selected_from_session}"
            },
            status=200,
        )

    except Exception as e:
        print(f"Exception in exp_is_weekly_selected :\n {e}")
        return JsonResponse(
            {
                "error": f"Exception in exp_is_weekly_selected : {is_weekly_selected_from_session}"
            },
            status=403,
        )


def date_check(request):
    try:
        data = pd.read_pickle(UPLOAD_FOLDER+"{}.pkl".format(request.session.get("_uuid")))
        # data = pd.DataFrame.from_dict(request.session.get('data'))
        # print(data)
        eo = explorer(data)
        print("submitted")
        DateSelector = request.GET.get("date_check")
        eo.date = DateSelector
        eo.date_check()
        print(DateSelector, "selected date")
        request.session["DateSelector"] = DateSelector
        request.session["cpm_checked"] = "False"
        # if request.session.get("cpm_checked") == True :
        #     print("deleting cpm session")
        #     del request.session['cpm_checked']
        print("date validated")
        return JsonResponse({"message": "Successfully published"}, status=200)

    except Exception as exp_date_check:
        return JsonResponse({"error": ERROR_DICT[str(exp_date_check)]}, status=403)
    

def dimension_grouping_check(request):
    try:
        data = pd.read_pickle(UPLOAD_FOLDER+"{}.pkl".format(request.session.get("_uuid")))
        # data = pd.DataFrame.from_dict(request.session.get('data'))
        # print(data)
        eo = explorer(data)
        print("submitted")
        dimension_grouping_check = request.GET.get('dimension_grouping_selector')
        dimensionSelector = request.session.get("DimensionSelector")
        eo.group_dimension = dimension_grouping_check if(len(dimensionSelector) > 1) else dimensionSelector[0]
        request.session["Dimension_grouping_check"] = dimension_grouping_check
        print("dimension grouping check validated")
        return JsonResponse({"message": "Successfully published"}, status=200)

    except Exception as exp_date_check:
        return JsonResponse({"error": ERROR_DICT[str(exp_date_check)]}, status=403)


def dimension_check(request):
    try:
        data = pd.read_pickle(UPLOAD_FOLDER+"{}.pkl".format(request.session.get("_uuid")))
        # data = pd.DataFrame.from_dict(request.session.get('data'))
        eo = explorer(data)
        print("submitted")
        # request.GET.getlist('fiel_name')
        DimensionSelector = request.GET.get("dimension_check[]")
        DimensionSelector = list(DimensionSelector.split(","))
        # print(DimensionSelector, type(DimensionSelector))
        DimensionSelector = DimensionSelector[:-1]
        eo.dimension = DimensionSelector
        eo.dimension_check()
        print(DimensionSelector, "selected dimension")
        request.session["DimensionSelector"] = DimensionSelector
        print("DimensionSelector", request.session.get("DimensionSelector"))
        print("dimension validated")
        return JsonResponse({"message": "Successfully published",
                             "dimensionSelector": DimensionSelector}, status=200)

    except Exception as exp_date_check:
        return JsonResponse({"error": ERROR_DICT[str(exp_date_check)]}, status=403)


def spent_check(request):
    try:
        # data = pd.DataFrame.from_dict(request.session.get('data'))
        data = pd.read_pickle(UPLOAD_FOLDER+"{}.pkl".format(request.session.get("_uuid")))
        eo = explorer(data)
        print("submitted")

        SpentSelector = request.GET.get("spent_check")
        eo.spend = SpentSelector
        eo.numeric_check("spend")
        print(SpentSelector, "selected spent")
        request.session["SpentSelector"] = SpentSelector
        return JsonResponse({"message": "Successfully published"}, status=200)

    except Exception as exp_date_check:
        return JsonResponse({"error": ERROR_DICT[str(exp_date_check)]}, status=403)


def target_check(request):
    try:
        # data = pd.DataFrame.from_dict(request.session.get('data'))
        data = pd.read_pickle(UPLOAD_FOLDER+"{}.pkl".format(request.session.get("_uuid")))
        eo = explorer(data)
        print("submitted")

        TargetSelector = request.GET.get("target_check")
        eo.target = TargetSelector
        eo.numeric_check("target")
        print(TargetSelector, "selected taret")
        request.session["TargetSelector"] = TargetSelector
        return JsonResponse({"message": "Successfully published"}, status=200)

    except Exception as exp_date_check:
        return JsonResponse({"error": ERROR_DICT[str(exp_date_check)]}, status=403)


def cpm_check(request):
    try:
        # data = pd.DataFrame.from_dict(request.session.get('data'))
        data = pd.read_pickle(UPLOAD_FOLDER+"{}.pkl".format(request.session.get("_uuid")))
        eo = explorer(data)
        print("submitted")

        CpmSelector = request.GET.get("cpm_check")
        if CpmSelector != "0":
            request.session["cpm_checked"] = "True"
            eo.use_impression = True
            eo.cpm = CpmSelector
            request.session["CpmSelector"] = CpmSelector
            eo.numeric_check("cpm")
        else:
            request.session["cpm_checked"] = "False"
            eo.use_impression = False
            eo.cpm = None
            request.session["CpmSelector"] = CpmSelector
 
        return JsonResponse({"message": "Successfully published"}, status=200)

    except Exception as exp_date_check:
        return JsonResponse({"error": ERROR_DICT[str(exp_date_check)]}, status=403)


def chart_filter(request):
    request.session["is_predict_page_submit_success"] = 0
    context = {}
    # data = pd.DataFrame.from_dict(request.session.get('line_chart_agg_data'))
    type = request.GET.get("type")
    chart_type = request.GET.get("chart_type")
    is_weekly_selected_from_session = request.session.get("is_weekly_selected")

    # start_date = datetime.strptime(request.GET.get("start_date"), "%m-%d-%y").date()
    # end_date = datetime.strptime(request.GET.get("end_date"), "%m-%d-%y").date()

    # start_date = datetime.strftime(start_date, "%m-%d-%y")
    # end_date = datetime.strftime(end_date, "%m-%d-%y")

    # print("*************Start Date***********************")
    # print(start_date, "strf")

    # print("*************End Date***********************")
    # print(end_date, "strf")

    if type == "submit":

        try:
            print("submit")

            data = pd.read_pickle(UPLOAD_FOLDER+"{}.pkl".format(request.session.get("_uuid")))
            print("DimensionSelector", request.session.get("DimensionSelector"))
            print("CpmSelector :", request.session.get("CpmSelector"))
            convert_to_weekly_data = request.session.get("convert_to_weekly_data")
            is_weekly_selected = request.session.get("is_weekly_selected")
            eo = explorer(data)
            eo.date = request.session.get("DateSelector")
            eo.dimension = request.session.get("DimensionSelector")
            eo.spend = request.session.get("SpentSelector")
            eo.target = request.session.get("TargetSelector")
            eo.cpm = request.session.get("CpmSelector")
            eo.group_dimension = request.session.get("Dimension_grouping_check")
            if convert_to_weekly_data != None and int(convert_to_weekly_data) == 1:
                eo.convert_to_weekly = True 
            if is_weekly_selected != None and int(is_weekly_selected) == 1:
                eo.is_weekly_selected = True 
            
            if request.session.get("cpm_checked") == "True":
                eo.use_impression = True
                print("use impressions true")
            # aggregation function
            (agg_data, dimension_data ) = eo.data_aggregation()
            request.session['dimension_data'] = dimension_data
            agg_data["date"] = pd.to_datetime(agg_data["date"])
            # print(agg_data.head(20))

            # get Start and End dates
            start_date = agg_data["date"].min()
            end_date = agg_data["date"].max()
            print("Start and End dates", start_date, end_date)

            # request.session['agg_data'] = agg_data.to_dict('dict')
            # agg_data.to_csv('sample_data/agg_data_test.csv')
            # agg_data = pd.DataFrame.from_dict(request.session.get('agg_data'))

            #  # ******************bar chart**********************
            print("agg_data", agg_data)
            agg_data["date"] = pd.to_datetime(agg_data["date"]).dt.date
            agg_data.to_pickle(
                UPLOAD_FOLDER+"{}_agg_data.pkl".format(request.session.get("_uuid"))
            )
            print("saved pickle file")

            agg_data_group = agg_data[
                (agg_data["date"] >= start_date) & (agg_data["date"] <= end_date)
            ]

            agg_data_group = agg_data_group.groupby("dimension").sum().reset_index()
            agg_data_group["CPA"] = (
                agg_data_group["spend"] / agg_data_group["target"]
            )  # creating new column CPA
            agg_data_group = agg_data_group.dropna().reset_index(drop=True)
            agg_data_group = agg_data_group[
                ~agg_data_group["CPA"].isin([-np.inf, np.inf])
            ]
            # # # ****************bar chart***********************

            # if default_dim == "0":
            #     default_dim = list(agg_data["dimension"].unique())[0]

            default_dim = list(agg_data["dimension"].unique())[0]
            unique_dim = list(agg_data["dimension"].unique())
            context["unique_dimension"] = unique_dim
            context["dimension_value_selector"] = list(
                set(unique_dim) - set([default_dim])
            )
            print("default_dim", default_dim)
            print("unique_dim", unique_dim)
            # dimension_value_selector = list(set(unique_dim) - set([default_dim]))

            agg_data_trendchart = agg_data[
                (agg_data["dimension"].isin(unique_dim))
                & (agg_data["date"] >= start_date)
                & (agg_data["date"] <= end_date)
            ].reset_index(drop=True).sort_values(by=["date"])
            agg_data_trendchart["date"] = pd.to_datetime(agg_data["date"]).dt.strftime(
                "%m-%d-%y"
            )  # datetime to string in that m-d-y format

            print("------- final filtered------------")
            print(agg_data.head())
            print("------- final filtered-------------")

            # aggregating the data 
            agg_data_trendchart = agg_data_trendchart.groupby([pd.Grouper(key='date')])['spend', 'target'].sum().reset_index().sort_values(['date'])
            # converting to dictionary for bar chart
            agg_data_group = agg_data_group.to_dict("records")
            # converting to dictionary for line chart
            agg_data_trendchart = agg_data_trendchart.to_dict("records")
            if is_weekly_selected_from_session:
                is_weekly_selected_from_session = int(is_weekly_selected_from_session)
                print(
                    f"is_weekly_selected_from_session : {is_weekly_selected_from_session}"
                )
                context[
                    "is_weekly_selected_from_session"
                ] = is_weekly_selected_from_session
            context["default_dim"] = default_dim
            context["chart_data_on_change"] = agg_data_trendchart
            context["bar_chart_data_on_change"] = agg_data_group  # bar chart data
            context["pie_chart_data_on_change"] = agg_data_group  # pie chart data
            context["start_date"] = start_date.strftime("%m/%d/%Y")
            context["end_date"] = end_date.strftime("%m/%d/%Y")

            # print(context['pie_chart_data_on_change'])

            return JsonResponse(context)
        except Exception as e:
            print("Exception in Submit", e)
    # ************************onchange**********************************
    elif type == "onchange":
        print("changed")

        start_date = datetime.strptime(request.GET.get("start_date"), "%m-%d-%y").date()
        end_date = datetime.strptime(request.GET.get("end_date"), "%m-%d-%y").date()

        print(
            "Start Date",
            start_date,
        )
        print("End Date", end_date)

        # DimensionValueSelector = request.GET.get('dimension_value_check[]') #dimension value selector from bar chart
        # print(DimensionValueSelector)

        # agg_data = pd.DataFrame.from_dict(request.session.get('agg_data'))
        agg_data = pd.read_pickle(
            UPLOAD_FOLDER+"{}_agg_data.pkl".format(request.session.get("_uuid"))
        )
        print("Data columns", agg_data.columns)
        # agg_data["date"] = pd.to_datetime(data["Day"]).dt.date # converting from string to datetime
        # for bar chart
        if chart_type == "linechart":
            # default_dim = request.GET.get("dimension_value_selector")
            default_dimensions = request.GET.get("dimension_value_selector").split(",")
            # print("Before default_dim", default_dim)
            #if default_dim == "0":
                #default_dim = list(agg_data["dimension"].unique())[0]
            # print("After default_dim", default_dim)

            unique_dim = list(agg_data["dimension"].unique())

            # context["dimension_value_selector"] = list(
            #     set(unique_dim) - set([default_dim])
            # )

            agg_data_trendchart = agg_data[
                (agg_data["dimension"].isin(default_dimensions))
                & (agg_data["date"] >= start_date)
                & (agg_data["date"] <= end_date)
            ]
            agg_data_trendchart["date"] = pd.to_datetime(agg_data["date"]).dt.strftime("%m-%d-%y")
            print("-------on change final filtered------------")
            print(agg_data.head())
            print("-------on change final filtered-------------")
            
            # aggregation
            agg_data_trendchart = agg_data_trendchart.groupby([pd.Grouper(key='date')])['spend', 'target'].sum().reset_index().sort_values(['date'])
            # data = data[data["dimension"]==default_dim].to_dict('records')
            agg_data_trendchart = agg_data_trendchart.to_dict("records")
            #context["default_dim"] = default_dim
            print("-------on change line chart defuat filtered-------------")
            context["chart_data_on_change"] = agg_data_trendchart

            return JsonResponse(context)
            print("onchange")
            # code for bar chart

        elif chart_type == "piechart":
            print("piechart")
            PieDimensionValueSelector = request.GET.get("pie_dimension_check[]")
            PieDimensionValueSelector = list(PieDimensionValueSelector.split(", "))
            print("PieDimensionValueSelector", PieDimensionValueSelector)
            # PieDimensionValueSelector = PieDimensionValueSelector[:-1]
            # print("this is pie dimension value selector", PieDimensionValueSelector)
            # ************************ Rerun this because dates from bar and pie chart might be different **********************************
            # agg_data = pd.DataFrame.from_dict(request.session.get('agg_data'))
            agg_data = pd.read_pickle(
                UPLOAD_FOLDER+"{}_agg_data.pkl".format(request.session.get("_uuid"))
            )
            # agg_data["date"] = pd.to_datetime(agg_data["date"]).dt.date # converting from string to datetime
            agg_data = agg_data[
                (agg_data["date"] >= start_date) & (agg_data["date"] <= end_date)
            ]  # date filter

            pie_agg_data_group = agg_data.groupby("dimension").sum().reset_index()
            pie_agg_data_group["CPA"] = (
                pie_agg_data_group["spend"] / pie_agg_data_group["target"]
            )  # creating new column CPA
            pie_agg_data_group = pie_agg_data_group[
                ~pie_agg_data_group["CPA"].isin([-np.inf, np.inf])
            ]
            pie_agg_data_group = pie_agg_data_group.dropna().reset_index(drop=True)
            pie_agg_data_group = pie_agg_data_group[
                pie_agg_data_group["dimension"].isin(PieDimensionValueSelector)
            ]
            print(
                "*******************pie chart on change*******************************"
            )
            print(pie_agg_data_group)
            print(
                "*******************pie chart on change*******************************"
            )
            pie_agg_data_group = pie_agg_data_group.to_dict(
                "records"
            )  # converting to dictionary for pie chart
            context["pie_chart_data_on_change"] = pie_agg_data_group  # bar chart data
            return JsonResponse(context)

        elif chart_type == "barchart":
            BarDimensionValueSelector = request.GET.get("bar_dimension_check[]")
            print(BarDimensionValueSelector)
            BarDimensionValueSelector = list(BarDimensionValueSelector.split(", "))
            print(BarDimensionValueSelector)
            # BarDimensionValueSelector = BarDimensionValueSelector[:-1]
            print(BarDimensionValueSelector, "this is bar dimension value selector")

            agg_data = pd.read_pickle(
                UPLOAD_FOLDER+"{}_agg_data.pkl".format(request.session.get("_uuid"))
            )
            # agg_data = pd.DataFrame.from_dict(request.session.get('agg_data'))
            agg_data["date"] = pd.to_datetime(
                agg_data["date"]
            ).dt.date  # converting from string to datetime
            agg_data_group = agg_data[
                (agg_data["date"] >= start_date) & (agg_data["date"] <= end_date)
            ]

            agg_data_group = agg_data_group.groupby("dimension").sum().reset_index()
            # creating new column CPA
            agg_data_group["CPA"] = agg_data_group["spend"] / agg_data_group["target"]
            agg_data_group = agg_data_group[
                ~agg_data_group["CPA"].isin([-np.inf, np.inf])
            ]
            agg_data_group = agg_data_group[
                agg_data_group["dimension"].isin(BarDimensionValueSelector)
            ]
            print(agg_data_group)
            agg_data_group = agg_data_group.to_dict(
                "records"
            )  # converting to dictionary for bar chart

            # for bar chart
            # print("*********************************on change data after date format change*********************************")
            # print(agg_data.head())
            # print(agg_data.dtypes)
            # print("*********************************on change data after date format change*********************************")
            # print("-------on change line chart filtered-------------")
            context["bar_chart_data_on_change"] = agg_data_group  # bar chart data
            return JsonResponse(context)

        return JsonResponse(context)


@login_required
def user_guide(request):
    try:
        return render(request, "home/user_guide.html")
    except Exception as e:
        raise e
