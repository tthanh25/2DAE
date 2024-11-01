import numpy as np
import PIL
from keras.models import load_model
import tensorflow as tf
from bm3d import bm3d
from skimage.metrics import structural_similarity as ssim
from PGD_attack import pgd

import matplotlib.pyplot as plt
from PIL import Image

mnist = tf.keras.datasets.mnist
(_, _), (x_test, _) = mnist.load_data()
x_test = np.reshape(x_test, (-1, 28, 28, 1))


def extract_parameters_color(model, clean_img, adv_img):
    SSIM = 0
    t = []
    #adv = np.array(PIL.Image.open(adv_img)) / 255
    #clean = np.array(PIL.Image.open(clean_img)) / 255
    c = np.reshape(clean, (1, 32, 32, 3))
    for r in np.arange(0, 1, 0.125):
        for g in np.arange(0, 1, 0.125):
            for b in np.arange(0, 1, 0.125):
                clean_est = bm3d_rgb(adv, [r, g, b])
                k = ssim(clean, clean_est,win_size=None, data_range=clean_est.max() - clean_est.min(), multichannel=True)
                clean_est = np.reshape(clean_est, (1, 32, 32, 3))
                if k > SSIM and np.argmax(model.predict(clean_est)) == np.argmax(model.predict(c)):
                    SSIM = k
                    t = [r, g, b]
    return t


def extract_parameters(model, clean_img, adv_img):
    SSIM = 0
    t = []

    # Normalize images
    adv = adv_img / 255.0
    clean = clean_img / 255.0

    # Convert to RGB if images are grayscale

    c = np.reshape(clean_img, (1, 28, 28, 1))
    
    for x in np.arange(0, 1, 0.125):
        clean_est = bm3d(adv, x)
        print(clean_est)

        clean_reshaped = clean.squeeze()
        print("clean_est số chiều:", clean_est.shape)
        print("clean số chiều:", clean.shape)
        print("adv số chiều:", adv.shape)
        k = ssim(clean_reshaped, clean_est, data_range=clean_est.max() - clean_est.min(), multichannel=True)
        print("k: ",k)
        clean_est = np.reshape(clean_est, (1, 28, 28, 1))
        if(k>SSIM):
         print(np.argmax(model.predict(clean_est)),np.argmax(model.predict(c)))
         if(np.argmax(model.predict(clean_est))==np.argmax(model.predict(c))):
          #print(r,g,b)
          SSIM=k
          t=[x]
    
    return t


model = load_model('train.h5')
c = True
k = 0

#for i in range(10000):
for i in range(10):
    clean = x_test[i]
    adv = pgd(clean, i)    
    t = extract_parameters(model, clean, adv)
    print(t)
    #print(adv)
    adv_reshaped = np.reshape(adv, (1, 28, 28, 1))  # Reshape adv for prediction
    prediction = model.predict(adv_reshaped)
    predicted_class = np.argmax(prediction)
    print(f"Prediction for adversarial image {i}: {predicted_class} with probabilities {prediction}")

    # Calculate and print accuracy
    true_class = np.argmax(model.predict(np.reshape(clean, (1, 28, 28, 1))))  # True class from clean image
    accuracy = (predicted_class == true_class).astype(float)  # Calculate accuracy for this prediction
    print(f"True class: {true_class}, Accuracy: {accuracy}")
    if t:
        np.reshape(t,(1))
        t = np.hstack((i, t))
        if c:
            v = t
            c = False
        else:
            v = np.vstack((v, t))
    
    if i % 50 == 0:
        if v.ndim == 1:
            v = v.reshape(-1, 1) 
        np.savez_compressed('data_prepare' + str(k), X=v[:, :1], Y=v[:, 1:])
        print(np.shape(v))