from flask import Flask, render_template, request, redirect, url_for
from keras.models import load_model
from keras.preprocessing import image
import cv2
import numpy as np
from tensorflow.keras import backend as K
from gradcam_utils import grad_cam
from integrated_gradients_utils import compute_and_visualize_integrated_gradients
from lime_utils_1 import compute_and_visualize_lime  # Replace with your actual function name


app = Flask(__name__)

dic = {0: 'Normal', 1: 'Doubtful', 2: 'Mild', 3: 'Moderate', 4: 'Severe'}
img_size = 224

# Custom F1 Score metric
def f1_score(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1_val


model = load_model('final_model-60.h5', custom_objects={'f1_score': f1_score})
model.make_predict_function()


def predict_label(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    colored = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)  # Convert grayscale to color
    resized = cv2.resize(colored, (img_size, img_size))
    i = image.img_to_array(resized) / 255.0
    i = i.reshape(1, img_size, img_size, 3)  # 3 channels for color
    p = np.argmax(model.predict(i), axis=-1)
    return dic[p[0]]


@app.route("/", methods=['GET', 'POST'])
def main():
    return render_template("index.html")


@app.route("/about")
def about_page():
    return "Please subscribe Artificial Intelligence Hub..!!!"


@app.route("/predict", methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        img = request.files['file']
        img_path = "uploads/" + img.filename
        img.save(img_path)

        # Make prediction
        p = predict_label(img_path)
        
        # Generate Grad-CAM visualization
        grad_cam(img_path, model, img_size=(img_size, img_size))

        # Compute and visualize integrated gradients
        compute_and_visualize_integrated_gradients(img_path, model)

        # Generate LIME explanation
        compute_and_visualize_lime(img_path, model)
        # lime_explanation = generate_lime_explanation(img_path, model)  # Replace with your actual function name
        # print(p, lime_explanation)
        
    print(p)
    return str(p).lower()


if __name__ == '__main__':
    app.run(debug=True)
