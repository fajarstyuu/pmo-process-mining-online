from django.urls import path
from .views import ConformanceCheckAPIView, IndexView, DiscoverModelAPIView

urlpatterns = [
    path('', IndexView.as_view(), name='index'),
    path('discover-model/', DiscoverModelAPIView.as_view(), name='discover-model'),
    path('conformance-check/', ConformanceCheckAPIView.as_view(), name='conformance-check'),
]