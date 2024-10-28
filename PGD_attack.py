import numpy as np
import tensorflow as tf
from cleverhans.tf2.attacks.projected_gradient_descent import projected_gradient_descent
from keras.models import load_model

# Load the model
model = load_model('train.h5')

def pgd(img, i):
    batch_shape = (1, 28, 28, 1)
    epsilon = [0.03, 0.06, 0.1, 0.2, 0.3, 0.4, 0.7]

    tf.get_logger().setLevel('INFO')
    
    # Preprocess the image
    img = (np.asarray(img) / 255.0).astype(np.float32)
    img = np.reshape(img, (1, 28, 28, 1))

    # Generate adversarial example using projected gradient descent
    adv_image = projected_gradient_descent(
        model,
        img,
        eps=epsilon[i % 7],
        eps_iter=0.01,
        nb_iter=40,
        norm=np.inf,  # Use np.inf for L-infinity norm
        clip_min=0.0,
        clip_max=1.0
    )
    adv_image = np.reshape(adv_image, (28, 28, 1))
    return adv_image  # Return the adversarial image