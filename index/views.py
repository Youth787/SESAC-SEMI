import base64
from django.shortcuts import render, get_object_or_404, redirect
from django.http import HttpResponseNotAllowed, StreamingHttpResponse
from django.utils import timezone
from django.core.paginator import Paginator
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render
from django.http import HttpResponse
import os
import cv2
from tensorflow.keras.models import load_model
from django.conf import settings
import numpy as np
from index.models import Image
import math
# from index.result import *


# Create your views here.
def main(request):
    return render(request, 'index/main.html')
def vgg_16(request):
    return render(request, 'index/vgg_16.html')
def contact(request):
    return render(request, 'index/contact.html')
def yolo(request):
    return render(request, 'index/yolo.html')


#########################
# vgg_16
#########################


@csrf_exempt
def save_snapshot(request):
    Image.objects.all().delete()
    if request.method == 'POST':
        image_data = request.POST['image_data']
        if image_data:
            image_data = image_data.replace('data:image/png;base64,', '')
            binary_data = base64.b64decode(image_data)
            
            print(f"Length of binary image data before decoding: {len(binary_data)}")


            # save the image to the database
            image_instance = Image(image_data=binary_data)
            image_instance.save()
            
            print(f"Saved image with ID: {image_instance.id}")
            
            saved_image = Image.objects.last()

            # Convert the binary image data to numpy array using np.frombuffer
            image_data = np.frombuffer(saved_image.image_data, dtype=np.uint8)

            # print the length of the binary image data after decoding
            print(f"Length of binary image data after decoding: {len(image_data)}")
            
            return JsonResponse({'status': 'ok'})
        else:
            return JsonResponse({'status': 'error', 'message': 'No image data provided'})
    else:
        return JsonResponse({'status': 'error', 'message': 'Invalid request method'})
    

    
model_path1 = os.path.join(settings.BASE_DIR, 'VGG_16_kfold_1.hdf5')
model1 = load_model(model_path1)
model_path2 = os.path.join(settings.BASE_DIR, 'VGG_16_kfold_2.hdf5')
model2 = load_model(model_path2)
model_path3 = os.path.join(settings.BASE_DIR, 'VGG_16_kfold_3.hdf5')
model3 = load_model(model_path3)
model_path4 = os.path.join(settings.BASE_DIR, 'VGG_16_kfold_4.hdf5')
model4 = load_model(model_path4)
model_path5 = os.path.join(settings.BASE_DIR, 'VGG_16_kfold_5.hdf5')
model5 = load_model(model_path5)



def upload(request):
    if request.method == "POST":
        # Load the saved image from the database
        saved_image = Image.objects.last() # Assuming you have at least one saved image
        
        print(f"Retrieved image with ID: {saved_image.id}")

        
        if saved_image:
            # Convert the binary image data to numpy array using np.frombuffer
            image_data = np.frombuffer(saved_image.image_data, dtype=np.uint8)
            
            # Decode the binary image data using cv2.imdecode
            img = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
            
            # Resize the image to 128x128 using cv2.resize
            img = cv2.resize(img, (128, 128))
            
            # Convert the PIL image to a NumPy array
            img_np = np.array(img)
            
            # Reshape the array and normalize pixel values
            img_np = img_np.reshape((1, 128, 128, 3))
            img_np = img_np.astype('float32') / 255

            # Print some information for debugging
            print(f"Image shape: {img.shape}")
            print(f"Image dtype: {img.dtype}")
            print(f"Image data range: {np.min(img)}, {np.max(img)}")
            
            # Load the model and make prediction

            pred_score = 0

            pred_score += model1.predict(img_np)
            pred_score += model2.predict(img_np)
            pred_score += model3.predict(img_np)
            pred_score += model4.predict(img_np)
            pred_score += model5.predict(img_np)
            pred_score = pred_score/5
            
            print("AVERAGE_SCORE :", pred_score)
            

            return render(request, "index/vgg_16.html", {"pred_score": pred_score})
        else:
            return HttpResponse("No saved image found")
    else:
        return HttpResponse("Invalid request method")





# def capture_face(request):
#     return StreamingHttpResponse(gen_frames(), content_type='multipart/x-mixed-replace; boundary=frame')



# def gen_frames():
#     video_capture = cv2.VideoCapture(0)
#     if not video_capture.isOpened():
#         raise Exception("Could not open video device")

#     while True:
#         ret, frame = video_capture.read()
#         if not ret:
#             break
        
#         frame =load_pathes(frame) #yolo original
        
#         ret, buffer = cv2.imencode('.jpg', frame)
#         frame = buffer.tobytes()

#         yield (b'--frame\r\n'
#                 b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
#         print("띠링")
