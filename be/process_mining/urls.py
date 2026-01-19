from django.urls import path
from .views import ConformanceCheckAPIView, IndexView, DiscoverModelAPIView, DiscoveryAPIView, ConformanceAPIView, DownloadModelAPIView, LoadEventLogAPIView, Discovery2APIView, StatisticAPIView, FilterAPIView, DownloadEventLogAPIView

urlpatterns = [
    path('', IndexView.as_view(), name='index'),
    path('discover-model/', DiscoverModelAPIView.as_view(), name='discover-model'),
    path('conformance-check/', ConformanceCheckAPIView.as_view(), name='conformance-check'),
    path('discover-model-2/', DiscoveryAPIView.as_view(), name='discover-model-2'),
    path('conformance-check-2/', ConformanceAPIView.as_view(), name='conformance-check-2'),
    path('download-model/', DownloadModelAPIView.as_view(), name='download-model'),
    path('load-event-log/', LoadEventLogAPIView.as_view(), name='load-event-log'),
    path('discover-model-3/', Discovery2APIView.as_view(), name='discover-model-3'),
    path('model-statistics/', StatisticAPIView.as_view(), name='model-statistics'),
    path('filter-event-log/', FilterAPIView.as_view(), name='filter-event-log'),
    path('download-event-log/', DownloadEventLogAPIView.as_view(), name='download-event-log'),
]