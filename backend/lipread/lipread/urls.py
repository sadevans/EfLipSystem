"""
URL configuration for lipread project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include
from django.conf.urls.static import static
from django.conf import settings
from lipread.core import views
from lipread.core.views import StreamHandler
from django.views.generic.base import RedirectView

# live_view = LiveView()
stream_handler = StreamHandler()

urlpatterns = [
    # path('', include('lipread.urls')),
    # path('admin/', admin.site.urls),
    path('', views.index, name='index'),
    path('upload_video/', views.upload_video, name='upload_video'),
    path('process_video/', views.process_video, name='provess_video'),
    path('get_video/', views.get_video, name='get_video'),
    # path('receive_frame/', views.receive_frame, name='receive_frame'),
    path('receive_frame/', stream_handler.receive_frame, name='receive_frame'),
    path('change_stream_flag/', stream_handler.change_stream_flag, name='change_stream_flag'),

    # path('favicon.ico', RedirectView.as_view(url=staticfiles_storage.url('favicon/favicon.ico')))
]
#   + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)