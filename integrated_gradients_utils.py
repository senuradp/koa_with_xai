import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K
import cv2

def compute_and_visualize_integrated_gradients(img_path, model):

    def integrated_gradients(input_image, model, steps=50):
        # Generate alphas
        alphas = tf.linspace(start=0.0, stop=1.0, num=steps+1)

        # Initialize TensorArray outside loop to collect gradients
        gradient_batches = tf.TensorArray(tf.float32, size=steps+1)

        # Iterate alphas range and batch computation for speed
        for alpha in tf.range(0, len(alphas), dtype=tf.float32):
            interpolated_image = input_image * alphas[tf.cast(alpha, tf.int32)]
            with tf.GradientTape() as tape:
                tape.watch(interpolated_image)
                preds = model(interpolated_image)
                target = preds[:, 0]
            gradients = tape.gradient(target, interpolated_image)
            gradient_batches = gradient_batches.write(tf.cast(alpha, tf.int32), gradients)
        
        print("Raw gradients:", gradients)


        # Stack path gradients together row-wise into single tensor
        total_gradients = gradient_batches.stack()

        # Integral approximation through averaging gradients
        avg_gradients = tf.reduce_mean(total_gradients, axis=0)

        # Scale integrated gradients with respect to input
        integrated_gradients = (input_image - 0.0) * avg_gradients

        return integrated_gradients


    # Load and preprocess an image
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Compute integrated gradients
    ig = integrated_gradients(img_array, model)

    # Normalize the integrated gradients for better visualization
    normalized_ig = (ig - tf.reduce_min(ig)) / (tf.reduce_max(ig) - tf.reduce_min(ig))

    # Convert TensorFlow tensor to NumPy array
    normalized_ig_numpy = normalized_ig[0].numpy()

    # Convert to 8-bit format for saving
    normalized_ig_uint8 = (normalized_ig_numpy * 255).astype('uint8')

    # Save the integrated gradients
    cv2.imwrite(f'integrated_gradient_outputs/integrated_gradients_{img_path.split("/")[-1]}', cv2.cvtColor(normalized_ig_uint8, cv2.COLOR_RGB2BGR))
