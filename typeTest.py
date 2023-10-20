from keras.models import load_model
import numpy as np  # Add this import
from image_utils import load_and_preprocess_image

# Load the multi-class tumor classification model
multi_model = load_model('MultiClassTumorModel.h5')

# Load the image for testing
image_path = 'D:\\datasets\\Testing\\pituitary\\Te-pi_0014.jpg'
input_img = load_and_preprocess_image(image_path)

if input_img is not None:
    multi_result = multi_model.predict(input_img)
    tumor_type = np.argmax(multi_result)

    if tumor_type == 0:
        print("Tumor Type: Glioma")
    elif tumor_type == 1:
        print("Tumor Type: Meningioma")
    elif tumor_type == 2:
        print("Tumor Type: Pituitary")
else:
    print("Failed to load the image.")
