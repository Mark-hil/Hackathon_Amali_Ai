# myapp/ml_model.py
import joblib
from PIL import Image
import numpy as np

# model = joblib.load('C:/Users/Chillop/Downloads/breast caner dataset/ImageClassification-main/savedModels/model.joblib')
# def preprocess_image(image):
#     # Resize image to 224x224
#     image = image.resize((224, 224))
#     # Convert image to numpy array
#     image = np.array(image)
#     # Check if the image has a single channel (grayscale)
#     if image.ndim == 2:
#         # Convert grayscale image to RGB by stacking
#         image = np.stack((image,)*3, axis=-1)
#     # Normalize the image
#     image = image / 255.0
#     # Reshape image to the shape your model expects
#     image = image.reshape(1, 224, 224, 3)
#     return image


# def predict(image):
#     processed_image = preprocess_image(image)
#     prediction = model.predict(processed_image)
#     return prediction

# myapp/ml_model.py
# import joblib
# from PIL import Image
# import numpy as np

# Load your trained model
# model = joblib.load('C:/Users/Chillop/Downloads/breast caner dataset/ImageClassification-main/savedModels/model.joblib')

# myapp/ml_model.py
import joblib
from PIL import Image
import numpy as np
# from tensorflow.keras.models import load_model

# model = joblib.load('E:/My Project Works/final yr/new one/New folder/model.h5')
# model = load_model('E:/My Project Works/final yr/new one/New folder/mobile.h5')

# def preprocess_image(image):
#     image = image.resize((128, 128))
#     image = np.array(image)
#     if image.ndim == 2:
#         image = np.stack((image,) * 3, axis=-1)
#     image = image / 255.0
#     image = image.reshape(1, 128, 128, 3) 
#     return image

# def predict(image):
#     processed_image = preprocess_image(image)
#     prediction = model.predict(processed_image)
#     print("Model prediction (probabilities):", prediction)
#     class_labels = ['positive', 'negative']
#     predicted_index = np.argmax(prediction[0])
#     predicted_label = class_labels[predicted_index]
#     print("Predicted label:", predicted_label)
#     return predicted_label

# Example usage
# image = Image.open('H:/Mark/DatasetMark2/negative')
# predicted_label = predict(image)