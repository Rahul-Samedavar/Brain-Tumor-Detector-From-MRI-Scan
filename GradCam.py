import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from tf_explain.core.grad_cam import GradCAM
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array


def imgArr(img_path):
    img = load_img(img_path, target_size=(299, 299))
    img = img_to_array(img)
    img /= 255
    return img, np.expand_dims(img, axis=0)

def grad_cam_plus(model, image, layer_name="block5_conv3", class_ind=None):
    conv_layer = model.get_layer(layer_name)
    TempModel = Model([model.inputs], [conv_layer.output, model.output])
    
    with tf.GradientTape() as gtape1:
        with tf.GradientTape() as gtape2:
            with tf.GradientTape() as gtape3:
                conv_output, predictions = TempModel(image)
                if class_ind is None:
                    class_ind = np.argmax(predictions[0])
                output = predictions[:, class_ind]
                first_grad = gtape3.gradient(output, conv_output)
            sec_grad = gtape2.gradient(first_grad, conv_output)
        ter_grad = gtape1.gradient(sec_grad, conv_output)

    summed = np.sum(conv_output[0], axis=(0, 1))
    num = sec_grad[0]
    den = sec_grad[0] * 2.0 + ter_grad[0] * summed
    den = np.where(den != 0.0, den, 1e-10)

    frac = num / den
    temp = np.sum(frac, axis=(0, 1))
    frac /= temp

    weights = np.maximum(first_grad[0], 0.0)
    vector = np.sum(weights * frac, axis=(0, 1))
    gc_map = np.sum(vector * conv_output[0], axis=2)

    heatmap = np.maximum(gc_map, 0)
    tresh = np.max(heatmap)
    tresh = max(tresh, 1e-10)
    heatmap /= tresh
    return heatmap



def mask(img_path, heatmap, alpha=0.4):
    img = cv2.imread(img_path)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = (heatmap * 255).astype("uint8")
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    imposed = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)
    return cv2.cvtColor(imposed, cv2.COLOR_BGR2RGB)


def grad_cam_visualize(model, image_path, layer, explainer=GradCAM()):
    classes = ['glioma', 'meningioma', 'notumor', 'pituitary']

    img, img_arr = imgArr(image_path)
    class_ind = np.argmax(model.predict(img_arr, verbose=0))
    explaination = explainer.explain((img_arr, None), model, class_ind, layer)
    fig = plt.figure()
    plt.subplot(1, 2, 1)
    plt.xlabel(f"{classes[class_ind]} Image", fontdict={'size': 12})
    plt.imshow(img)
    plt.subplot(1, 2, 2)
    plt.xlabel("Grad Cam Visualization", fontdict={'size': 12})
    plt.imshow(explaination)
    return fig

def grad_cam_plus_plus_visualize(model, image_path, layer, alpha=0.3):
    classes = ['glioma', 'meningioma', 'notumor', 'pituitary']

    img, img_arr = imgArr(image_path)
    class_ind = np.argmax(model.predict(img_arr, verbose=0))

    heatmap = grad_cam_plus(model, img_arr, layer, class_ind)
    masked = mask(image_path, heatmap, alpha)
    fig = plt.figure()
    plt.subplot(1, 2, 1)
    plt.xlabel(f"{classes[class_ind]} Image", fontdict={'size': 12})
    plt.imshow(img)
    plt.subplot(1, 2, 2)
    plt.xlabel("Grad Cam++ Visualization", fontdict={'size': 12})
    plt.imshow(masked)
    return fig