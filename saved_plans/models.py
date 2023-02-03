from django.db import models
from django.contrib.auth.models import User
# # Create your models here.
class SavedPlan(models.Model):
    plan_id = models.BigAutoField(primary_key=True)
    plan_name = models.CharField(max_length=30, null=True)
    plan_date = models.DateField()
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    # user_email = models.OneToOneField(Users, on_delete=models.CASCADE,verbose_name="related place",)
    plan_result_donut_chart_path = models.TextField()
    plan_result_table_path = models.TextField()
    left_hand_panel_data_path = models.TextField()
    discarded_dimensions_data_path = models.TextField()

    def __str__(self):
        return self.plan_name
