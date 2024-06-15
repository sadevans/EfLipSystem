from django import forms
from .models import VideoFile

class VideoUploadForm(forms.ModelForm):
    # model = Video
    class Meta:
        model = VideoFile
        fields = ('video_file',)
    # video_file = forms.FileField(label="Upload video", required=False)