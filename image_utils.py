import cv2
from PIL import Image
import numpy as np

def load_and_preprocess_image(image_path, target_size=(64, 64)):
    image = cv2.imread(image_path)

    if image is not None:
        img = Image.fromarray(image)
        img = img.resize(target_size)
        img = np.array(img)

        input_img = np.expand_dims(img, axis=0)
        return input_img
    else:
        return None
