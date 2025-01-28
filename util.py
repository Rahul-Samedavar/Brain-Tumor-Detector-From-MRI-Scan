import numpy as np
from PIL import Image

classes = ['glioma', 'meningioma', 'notumor', 'pituitary']

def get_probs(img_path, model):
    img = Image.open(img_path)

    resized_img = img.resize((299, 299))
    if img.mode != 'RGB':
        img = np.stack([np.array(resized_img)] * 3, axis=-1)
    else:
        img = np.asarray(resized_img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    predictions = model.predict(img, verbose=0)
    return list(predictions[0])

def predict(img_path, model):
    probs = get_probs(img_path, model)
    return np.argmax(probs)

def classify(img_path, model):
    return classes[predict(img_path, model)]