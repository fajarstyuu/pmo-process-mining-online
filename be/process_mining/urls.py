from django.urls import path
from .views import IndexView, DiscoverModelAPIView

urlpatterns = [
    path('', IndexView.as_view(), name='index'),
    path('discover-model/', DiscoverModelAPIView.as_view(), name='discover-model'),
]