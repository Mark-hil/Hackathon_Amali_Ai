from django.urls import path

from djangoBreastDetection import settings
from . import views
from .views import home, predict
# from .views import home,chatbot
from django.conf.urls.static import static

urlpatterns = [
    path('', views.Welcome, name = 'Welcome'),
    # path('user', views.User, name = 'User')
    path('', home, name='home'),
    path('upload/', predict, name='upload_image'),
    path('predict/', predict, name='predict'), 
    # path('chatbot/', views.chatbot, name='chatbot'),
]+ static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
