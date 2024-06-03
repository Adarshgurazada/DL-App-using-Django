from django.shortcuts import render
import numpy as np
from django.core.files.storage import FileSystemStorage
from tensorflow.keras.models import load_model
from .functions import preprocess_image
import cv2
from django.contrib.auth.decorators import login_required

# Assuming the model file is saved in the dl_model directory
model_path = 'dl_model/vgg19_model.h5'
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'] 


@login_required
def dl_model_view(request):
    if request.method == 'POST' and request.FILES['image']:
        uploaded_image = request.FILES['image']
        # Convert uploaded image to a NumPy array
        try:
            nparr = np.frombuffer(uploaded_image.read(), np.uint8)
            img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # Assuming it's a color image
            # Preprocess the uploaded image
            processed_image = preprocess_image(img_np)  # Assuming preprocess_image expects a NumPy array
            # Add batch dimension to the processed image
            processed_image_batch = np.expand_dims(processed_image, axis=0)
            # Load the saved model
            model = load_model(model_path)
            # Perform prediction
            prediction = model.predict(processed_image_batch)
            # Get the predicted class name (assuming class_names is defined)
            predicted_class = class_names[np.argmax(prediction)]
            return render(request, 'dl_model.html', {'uploaded_image': uploaded_image, 'predicted_class': predicted_class})
        except Exception as e:
            print(f"Error processing uploaded image: {e}")
            return render(request, 'dl_model.html', {'error_message': 'Error processing uploaded image'})
    return render(request, 'dl_model.html')

# def dl_model_view(request):
#     if request.method == 'POST' and request.FILES['image']:
#         uploaded_image = request.FILES['image']
#         fs = FileSystemStorage()
#         image_path = fs.save(uploaded_image.name, uploaded_image)
#         # Preprocess the uploaded image
#         processed_image = preprocess_image(image_path)
#         # Load the saved model
#         model = load_model(model_path)
#         # Perform prediction
#         prediction = model.predict(processed_image)
#         # Get the predicted class name (assuming class_names is defined)
#         predicted_class = class_names[np.argmax(prediction)]
#         return render(request, 'dl_model.html', {'uploaded_image': image_path, 'predicted_class': predicted_class})
#     return render(request, 'dl_model.html')


# def dl_model_view(request):
#     if request.method == 'POST' and request.FILES['image']:
#         uploaded_image = request.FILES['image']
#         # Convert uploaded image to a NumPy array
#         try:
#             nparr = np.frombuffer(uploaded_image.read(), np.uint8)
#             img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # Assuming it's a color image
#             # Preprocess the uploaded image
#             processed_image = preprocess_image(img_np)  # Assuming preprocess_image expects a NumPy array
#             # Load the saved model
#             model = load_model(model_path)
#             # Perform prediction
#             prediction = model.predict(processed_image)
#             # Get the predicted class name (assuming class_names is defined)
#             predicted_class = class_names[np.argmax(prediction)]
#             return render(request, 'dl_model.html', {'uploaded_image': uploaded_image, 'predicted_class': predicted_class})
#         except Exception as e:
#             print(f"Error processing uploaded image: {e}")
#             return render(request, 'dl_model.html', {'error_message': 'Error processing uploaded image'})
#     return render(request, 'dl_model.html')



