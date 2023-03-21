from django.shortcuts import render, redirect
from django.contrib import messages
from .forms import UserRegisterForm
from .tokens import account_activation_token
from django.template.loader import render_to_string
from django.contrib.sites.shortcuts import get_current_site
from django.utils.http import urlsafe_base64_encode, urlsafe_base64_decode
from django.utils.encoding import force_bytes, force_str
from django.core.mail import EmailMessage

def activateEmail(request, user, to_email, username):
    mail_subject = 'Activate your user account.'
    message = render_to_string('users/template_activate_account.html', {
        'user': user,
        'domain': get_current_site(request).domain,
        'uid': urlsafe_base64_encode(force_bytes(user.pk)),
        'token': account_activation_token.make_token(user),
        'protocol': 'https' if request.is_secure() else 'http'
    })
    email = EmailMessage(mail_subject, message, to=[to_email])
    if email.send():
        messages.success(request, f'please check your email {to_email} inbox and complete \
           registration process ')
    else:
        messages.error(request, f'Problem sending confirmation email to {to_email}, check if you typed it correctly.')

def activate(request, uidb64, token):
    User = UserRegisterForm.get_user_model()
    try:
        uid = force_str(urlsafe_base64_decode(uidb64))
        user = User.objects.get(pk=uid)
    except(TypeError, ValueError, OverflowError, User.DoesNotExist):
        user = None

    if (user is not None and account_activation_token.check_token(user, token)) or user.is_active == True:
        user.is_active = True
        user.save()

        messages.success(request, 'Thank you for your email confirmation. Now you can login your account.')
        return redirect('/login')
    else:
        messages.error(request, 'Activation link is invalid!')
    
    return redirect('/login')

def register(request):
    if request.method == "POST":
        form = UserRegisterForm(request.POST)
        if form.is_valid():
            # commit= False , means it will not be saved in database 
            user = form.save(commit=False)
            # this means user can't login without email activation
            user.is_active = False
            user.save()
            username = form.cleaned_data.get("username")
            activateEmail(request, user, form.cleaned_data.get('email'), username )
           
            # messages.success(
            #     request,
            #     f"{username} : your account has been created! You are now able to log in",
            # )
    else:
        form = UserRegisterForm()
    return render(request, "users/register.html", {"form": form})