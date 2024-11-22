# Breast_Detection_with-_Django

This application is designed to aid in the detection and classification of breast cancer using machine learning models. Built with Django as the backend framework, the app provides a streamlined user experience for healthcare practitioners and researchers.

# Features
Image Upload: Users can upload medical images for analysis.
Cancer Detection: The app uses a pre-trained deep learning model to classify images into benign or malignant categories.
Detailed Results: After processing, the app displays the probability and confidence level for each classification.
User Management: Secure authentication and user management for personalized access and record tracking.

# Getting Started
# 1. Prerequisites
Python 3.8 or above
Django 3.0 or above
TensorFlow / Keras (for model integration)
Other dependencies specified in requirements.txt

# 2. Installation
Clone the repository:
git clone https://github.com/Mark-hil/Breast_Detection_with-_Django.git

# 3. Navigate to the project directory:
cd breast-cancer-app

# 4. Install the required dependencies:
pip install -r requirements.txt
Run migrations:
python manage.py migrate
Start the development server:
python manage.py runserver

# Model Information
The app uses a deep learning model trained on public breast cancer datasets (e.g., BUSI, IDC_regular_ps50_idx5) to classify images into benign and malignant categories.
# Hackathon_Amali_Ai
