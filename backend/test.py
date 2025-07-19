from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

model = load_model('./models/brain_tumor_detector_model.h5')

class_labels = ['glioma', 'meningioma', 'notumor', 'pituitary']

img_path = './test_images/sample_image3.jpg'  
img = image.load_img(img_path, target_size=(150, 150))
img = image.img_to_array(img)
img = np.expand_dims(img, axis=0) / 255.0

prediction = model.predict(img)
predicted_class = np.argmax(prediction[0])
confidence = prediction[0][predicted_class]

if class_labels[predicted_class] == 'notumor':
    print("No tumor detected.")
else:
    print(f"Tumor type detected: {class_labels[predicted_class]}")
print(f"Confidence: {confidence * 100:.2f}%")