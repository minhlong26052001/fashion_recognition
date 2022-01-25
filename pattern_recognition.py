import cv2 
import numpy as np 


def get_pattern(img, model, class_names):
    img_resized = cv2.resize(img, (224,224))
    img_input = np.expand_dims(img_resized, axis=0)
    output = model.predict(img_input)[0]
    max_index = np.argmax(output)
    pattern = class_names[max_index]

    return pattern