from django import forms
from django.contrib.auth.models import User
from django.contrib.auth.forms import UserCreationForm


class UserRegisterForm(UserCreationForm):
    email = forms.EmailField()
    role = forms.CharField()
    company = forms.CharField()
    def get_user_model():
        return User
    def clean_email(self):
        data = self.cleaned_data['email']
        if User.objects.filter(email=data).exists():
            raise forms.ValidationError("This email already used")
        elif "@xmedia.com" not in data:   
            raise forms.ValidationError("Must be Xmedia Email Address")
        return data

    class Meta:
        model = User
        fields = ["username", "email", "role", "company", "password1", "password2"]
        
