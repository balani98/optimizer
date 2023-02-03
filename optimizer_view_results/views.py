from django.shortcuts import render
from saved_plans.models import SavedPlan
from django.http import HttpResponse, JsonResponse
from django.core import serializers
import json
from django.core.serializers.json import DjangoJSONEncoder
import boto3
# Create your views here.

def optimizer_view_results(request):
    try:
        plan_id = request.GET.get('id')
        context = {}
        context['plan_id'] = plan_id
        return render(request, "optimizer_view_results.html",context)
    except Exception as e:
        raise e

def get_object_from_S3(path_to_file) :
    json_config_data = ""
    with open('config.json') as config_file:
        json_config_data = json.load(config_file)
    session = boto3.Session(
        aws_access_key_id = json_config_data['ACCESS_KEY_ID'],
        aws_secret_access_key = json_config_data['SECRET_ACCESS_KEY'],
    )
    s3 = session.resource('s3')
    content_object = s3.Object('optimizer-bkt', path_to_file)
    file_content = content_object.get()['Body'].read().decode('utf-8')
    json_content = json.dumps(file_content)
    return json_content


def SavedPlanToDictionary(savedplan):
    if savedplan is None:
        return None
    plan_result_table = get_object_from_S3(savedplan[0]["plan_result_table_path"])
    plan_result_donut_chart_data = get_object_from_S3(savedplan[0]["plan_result_donut_chart_path"])
    dictionary = {}
    dictionary["plan_id"] = savedplan[0]['plan_id']
    dictionary["plan_name"] = savedplan[0]['plan_name']
    dictionary["plan_result_table"] = plan_result_table
    dictionary["plan_result_donut_chart"] = plan_result_donut_chart_data
    return dictionary 


def table_and_chart_results(request):
    try:
        plan_id = int(request.GET.get('plan_id'))
        saved_plan = SavedPlan.objects.filter(plan_id=plan_id).values()
        
        saved_plan_dict = SavedPlanToDictionary(saved_plan)
        return JsonResponse(saved_plan_dict, status=200)
    except Exception as e:
        print(str(e))
        return HttpResponse(str(e), status=403)