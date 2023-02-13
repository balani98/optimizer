"""
WSGI config for x_media_optimizer_django_project project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/4.0/howto/deployment/wsgi/
"""

import os
from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'x_media_optimizer_django_project.settings')
os.environ.setdefault('ENVIRONMENT','test')

application = get_wsgi_application()
