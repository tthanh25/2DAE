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
                k = ssim(clean, clean_est, data_range=clean_est.max() - clean_est.min(), multichannel=True)
                clean_est = np.reshape(clean_est, (1, 32, 32, 3))
                if k > SSIM and np.argmax(model.predict(clean_est)) == np.argmax(model.predict(c)):
                    SSIM = k
                    t = [r, g, b]
    return t


def extract_parameters(model, clean_img, adv_img):
    SSIM = 0
    t = []

    clean = np.array(clean_img) / 255.0  # Normalize clean image
    adv = np.array(adv_img) / 255.0  # Normalize adversarial image
    c = np.reshape(clean_img, (1, 28, 28, 1))
    for x in np.arange(0, 1, 0.125):
        clean_est = bm3d_rgb(adv, x)
        k = ssim(clean_img, clean_est, data_range=clean_est.max() - clean_est.min(), multichannel=True)
        clean_est = np.reshape(clean_est, (1, 28, 28, 1))
        if k > SSIM and np.argmax(model.predict(clean_est)) == np.argmax(model.predict(c)):
            SSIM = k
            t = [x]
    return t


model = load_model('train.h5')
c = True
k = 0

for i in range(10000):
    clean = x_test[i]
    print(clean)
    adv = pgd(clean, i)
    print("adv pgd:")
    print(adv)
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.title('ANH NHIEUUUUUUUUUUUUUU')
    plt.axis('off')
    plt.imshow(adv)
    plt.subplot(1, 3, 1)
    plt.title('ANH GOCCCCCCCCCCCC')
    plt.axis('off')
    plt.imshow(clean)
    t = extract_parameters(model, clean, adv)

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