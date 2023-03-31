from django.contrib import admin
from django.urls import path, include


urlpatterns = [
    path('admin/', admin.site.urls),
    path('common/', include('common.urls')),
    path('', include('index.urls')),
    # path('', views.main, name='main'),  # '/' 에 해당되는 path
]