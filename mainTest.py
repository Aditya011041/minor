from keras.models import load_model
from image_utils import load_and_preprocess_image

# Load the binary classification model (tumor yes or no)
model = load_model('BrainTumor10EpochsCategorical.keras')

# Load the image for testing
image_path = 'D:\\brain\\pred\\pred49.jpg'
input_img = load_and_preprocess_image(image_path)

if input_img is not None:
    result = model.predict(input_img)

    if result[0][1] > 0.5:
        print("Tumor Detected")
    else:
        print("No Tumor Detected")
else:
        print("Failed to load the image.")
