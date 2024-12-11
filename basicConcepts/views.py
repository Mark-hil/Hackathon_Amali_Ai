from django.shortcuts import redirect, render
from django.http import HttpResponse
import numpy as np
import tensorflow as tf
from django.shortcuts import render
from django.http import JsonResponse
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
from django.core.files.storage import FileSystemStorage
from django.shortcuts import render
from django.http import JsonResponse
from django.shortcuts import render
from PIL import Image
import numpy as np
import io
import base64
from tensorflow.keras.models import load_model
from djangoBreastDetection import settings
from .forms import ImageUploadForm
from PIL import Image
from django.views.decorators.csrf import csrf_exempt
def Welcome(request):
    return render(request, 'index.html')


def home(request):
    return redirect('predict')


#model = load_model('E:/My Project Works/final yr/new one/Mark/mobNew.h5')

# Load the Keras model from the .h5 file
model = load_model('breast-cancer-detection-/mobile.h5')

# Define the class labels
class_labels = ["Benign", "Malignant"]

from django.core.files.storage import FileSystemStorage

# Define the class labels
class_labels = ["Benign", "Malignant"]


from django.http import JsonResponse
from django.core.files.storage import FileSystemStorage
from PIL import Image
import numpy as np
from keras.preprocessing.image import img_to_array
from django.views.decorators.csrf import csrf_exempt


@csrf_exempt
def predict(request):
    predicted_class_label = None  # Initialize the variable with a default value
    predicted_probability = None  # Initialize the probability
    image_path = None  # Initialize image_path to avoid referencing before assignment
    
    if request.method == 'POST' and request.FILES.get('image'):
        uploaded_file = request.FILES['image']

        # Save the uploaded file to the media directory
        # fs = FileSystemStorage()
        # filename = fs.save(uploaded_file.name, uploaded_file)
        # image_path = fs.url(filename)

        try:
            # Open the image using PIL (Pillow)
            image = Image.open(uploaded_file)

            # Preprocess the image
            image = image.resize((128, 128))  # Resize to model's expected input size
            image_array = img_to_array(image) / 255.0  # Convert to array and normalize
            image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

            # Perform prediction using the Keras model
            prediction = model.predict(image_array)

            # Get the predicted class index (0 or 1 for binary classification)
            predicted_class_index = np.argmax(prediction, axis=1)[0]
            predicted_class_label = class_labels[predicted_class_index]

            # Get the predicted probability for the predicted class
            predicted_probability = round(float(prediction[0][predicted_class_index]), 2)  # Round to 2 decimal places

        except Exception as e:
            return JsonResponse({
                'error': f"Error during processing: {str(e)}"
            }, status=500)

        # Return the prediction result in JSON format
        return JsonResponse({
            'prediction': predicted_class_label,
            'probability': predicted_probability,
            'image_path': image_path
        })
    else:
        # If it's a GET request, render the upload form
        return render(request, "upload.html")


