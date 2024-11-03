import numpy as np
import PIL
from keras.models import load_model
import tensorflow as tf
from bm3d import bm3d
from skimage.metrics import structural_similarity as ssim
from PGD_attack import pgd

mnist = tf.keras.datasets.mnist
(_, _), (x_test, _) = mnist.load_data()
x_test = np.reshape(x_test, (-1, 28, 28, 1))

def extract_parameters_color (model,clean_img,adv_img):
  SSIM=0
  t=[]
  adv = np.array(PIL.Image.open(adv_img)) / 255
  clean = np.array(PIL.Image.open(clean_img)) / 255
  c=np.reshape(clean,(1,32,32,3))
  for r in np.arange(0,1,0.125):
    for g in np.arange (0,1,0.125):
      for b in np.arange (0,1,0.125):
        clean_est = bm3d_rgb(adv, [r,g,b])
        k=ssim(clean,clean_est,data_range=clean_est.max() - clean_est.min(),multichannel=True)
        clean_est=np.reshape(clean_est,(1,32,32,3))
        if(k>SSIM):
         #print(np.argmax(model.predict(clean_est)),np.argmax(model.predict(c)))
         if(np.argmax(model.predict(clean_est))==np.argmax(model.predict(c))):
          #print(r,g,b)
          SSIM=k
          t=[r,g,b]
  return t

  
def extract_parameters(model, clean_img, adv_img):
    SSIM = 0
    t = 0

    c = np.reshape(clean_img, (1, 28, 28, 1))
    clean_reshaped = clean_img.squeeze()
    
    for x in np.arange(0, 1, 0.125):
        clean_est = bm3d(adv_img, x)
        k = ssim(clean_reshaped, clean_est, data_range=clean_est.max() - clean_est.min(), multichannel=True)
        print("k: ",k)
        clean_est = np.reshape(clean_est, (1, 28, 28, 1))
        if k > SSIM:
          print(np.argmax(model.predict(clean_est)),np.argmax(model.predict(c)))
          if(np.argmax(model.predict(clean_est))==np.argmax(model.predict(c))):
            SSIM = k
            t = x  # Store the current x value where SSIM is maximum
    
    return t  # Return x value


model = load_model('train.h5')
# Initialize lists for data collection
v = []
t_values = []

for i in range(10000):
    clean = x_test[i]
    adv = pgd(clean, i)    
    t = extract_parameters(model, clean, adv)
    print(t)
    
    adv_reshaped = np.reshape(adv, (1, 28, 28, 1))  # Reshape adv for prediction
    prediction = model.predict(adv_reshaped)
    predicted_class = np.argmax(prediction)
    true_class = np.argmax(model.predict(np.reshape(clean, (1, 28, 28, 1))))
    accuracy = (predicted_class == true_class).astype(float)
    print(f"Prediction for adversarial image {i}: {predicted_class} with probabilities {prediction}")
    print(f"True class: {true_class}, Accuracy: {accuracy}")
    
    if t is not None:  # Ensure t is a valid value
        v.append(i)  # Collect index
        t_values.append(t)  # Collect the corresponding t value

    # Save every 50 iterations into the same file
    if i % 50 == 0:
        np.savez_compressed('data_prepare', X=np.array(v), Y=np.array(t_values))
        print(f"Saved data at iteration {i}: {np.shape(v)}, {np.shape(t_values)}")

# Final save after loop completion
np.savez_compressed('data_prepare_final', X=np.array(v), Y=np.array(t_values))