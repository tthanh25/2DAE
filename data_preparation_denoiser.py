import numpy as np
import PIL
from keras.models import load_model
import tensorflow as tf
from bm3d import bm3d_rgb
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

    #print("adv trong ham extract")
    #print(adv)

    # Convert to RGB if images are grayscale
    if adv.shape[-1] == 1:
        adv = np.repeat(adv, 3, axis=-1)
    if clean.shape[-1] == 1:
        clean = np.repeat(clean, 3, axis=-1)

    c = np.reshape(clean_img, (1, 28, 28, 1))
    
    for x in np.arange(0, 1, 0.125):
        clean_est = bm3d_rgb(adv, x)
        
        # Kiểm tra giá trị
        if np.any(np.isnan(clean_est)) or np.any(np.isinf(clean_est)):
            print("clean_est contains NaN or inf values")
            continue
        
        #print("clean_est số chiều:", clean_est.shape)
        
        # Kiểm tra kích thước
        if clean.shape[0] < 7 or clean.shape[1] < 7 or clean_est.shape[0] < 7 or clean_est.shape[1] < 7:
            print("Một trong các ảnh quá nhỏ!")
            continue
        
        clean_est = np.clip(clean_est, 0, 1)  # Ensure values are within [0, 1]
        
        k = ssim(clean, clean_est, data_range=clean_est.max() - clean_est.min(), multichannel=True, win_size=3)

        clean_est = np.reshape(clean_est, (1, 28, 28, 1))
        if k > SSIM and np.argmax(model.predict(clean_est)) == np.argmax(model.predict(c)):
            SSIM = k
            t = [x]
    
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
    if t:
        t = np.hstack((i, t))
        if c:
            v = t
            c = False
        else:
            v = np.vstack((v, t))
    
    if i % 50 == 0 and not c:
        np.savez_compressed('data_prepare' + str(k), X=v[:, :1], Y=v[:, 1:])
        print(np.shape(v))
        k += 1