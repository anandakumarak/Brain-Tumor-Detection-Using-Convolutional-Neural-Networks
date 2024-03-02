import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the saved model
model = load_model('D:\\PROJECTS\\BRAIN_TUMOR\\BrainTumorDetection.h5')

# Load and preprocess the input image
img_path = 'D:\\PROJECTS\\BRAIN_TUMOR\\data\\tumor_detected\\y2.jpg'  # Replace with the path to your image
img = image.load_img(img_path, target_size=(64, 64))
img = image.img_to_array(img)
img = np.expand_dims(img, axis=0)
img = img / 255.0

# Get the model's prediction
prediction = model.predict(img)

if prediction[0] >= 0.5:
    prediction_label = "Yes, Tumor detected"
else:
    prediction_label = "No, Tumor not detected"

# Display the image with predicted output
plt.imshow(img[0])

if prediction_label=="Yes, Tumor detected":
    plt.text(0, -6, prediction_label, color='black', backgroundcolor='red', fontsize=12)

else:
    plt.text(0, -6, prediction_label, color='black', backgroundcolor='green', fontsize=12)

plt.axis('off')
plt.show()

