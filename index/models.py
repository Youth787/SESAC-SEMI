from django.db import models

class Image(models.Model):
    image_data = models.BinaryField()
    class Meta:
        app_label = 'index'