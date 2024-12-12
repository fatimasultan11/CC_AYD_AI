from django.urls import path

# from . import viewss_whisper
from . import whisperxView
from . import views

urlpatterns = [
    path("", views.index, name="index"),
    # path('whisper_analysis/', viewss_whisper.whisper_analysis_api, name='whisper_analysis')
    path('whisper_analysis/', whisperxView.process_audio, name='whisper_analysis')
]
