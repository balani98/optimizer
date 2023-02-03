from django.contrib.auth.decorators import login_required
from django.http import JsonResponse,HttpResponse
from django.shortcuts import render
from saved_plans.models import SavedPlan
# Create your views here.


@login_required
def saved_plan_list(request):
    try:
        current_user = request.user
        saved_plans_list = SavedPlan.objects.filter(user=current_user).values('plan_id', 'plan_name', 'plan_date')
        return render(request, "saved_plans/saved_plans_list.html",{'saved_plans_list':saved_plans_list})
    except Exception as e:
        raise e


def delete_saveplan(request):
    try:
        plan_id = request.GET.get('plan_id')
        saved_plan = SavedPlan.objects.filter(plan_id=plan_id)
        saved_plan.delete()
        return JsonResponse({'message': 'plan deleted successfully'}, status=200)
    except Exception as error:
        return HttpResponse(str(error), status=500)
