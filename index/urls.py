from django.urls import path

from . import views
app_name = 'index'

urlpatterns = [
    path('', views.main, name='main'),
    path("contact/", views.contact, name="contact"),
    path("vgg_16/", views.vgg_16, name="vgg_16"),
    # path("yolo/", views.yolo, name="yolo"),
  
    path('vgg_16/save-snapshot/', views.save_snapshot, name='save_snapshot'),
    path("upload/", views.upload, name="upload"),
    
    # path('capture_face/', views.capture_face, name='capture_face'),
]