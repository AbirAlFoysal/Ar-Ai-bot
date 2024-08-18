from django.urls import path
from . import views

urlpatterns = [
    path('', views.chat, name='chat'),
    # path('emotion/', views.second_page, name='emotion'),
    path('emotion/', views.video_feed, name='emotion'),
    path('submit_picture/', views.submit_picture, name='submit_picture'),
    path('submit_voice/', views.submit_voice, name='submit_voice'),
    path('get_video/', views.get_video, name='get_video'),
    # path('get_video/', views.get_video, name='get_video'),
    path('get_video/', views.get_video, name='get_video'),
    path('stream_video/', views.stream_video, name='stream_video'),


]
