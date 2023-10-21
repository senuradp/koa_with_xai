import tensorflow as tf

# Disable eager execution (required for LIME)
# tf.compat.v1.disable_eager_execution()

from tensorflow.keras import backend as K
from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
import cv2


# Custom F1 Score metric
def f1_score(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1_val

def compute_and_visualize_lime(image_path, model):

    # Load the saved model
    loaded_model = model

    # Load a specific image (replace with your image path)
    image_path = image_path

################################################ Lime Visualization ################################################ 

    # Label mapping
    label_mapping = {
        0: 'Normal',
        1: 'Doubtful',
        2: 'Mild',
        3: 'Moderate',
        4: 'Severe'
    }

    # Data directories and parameters
    img_size = 224
    batch_size = 32

    # Data normalization
    test_datagen = ImageDataGenerator(rescale=1./255)

    # Initialize LIME explainer
    explainer = lime_image.LimeImageExplainer()

    # 
    image = Image.open(image_path).convert('L')

    # Resize the image to the target size (224, 224)
    image = image.resize((img_size, img_size))

    # Convert to numpy array and normalize
    image_array = np.array(image) / 255.0

    # If the image is grayscale, stack it to make it 3-channel for the model
    image_array_rgb = np.stack([image_array]*3, axis=-1)

    # Expand dimensions to represent a batch of size 1
    image_batch_rgb = np.expand_dims(image_array_rgb, axis=0)

    # Use this image for prediction and explanation
    test_img = np.expand_dims(image_array, axis=0)

    # Prediction function for LIME
    def model_predict(img):
        if img.shape[-1] == 1 or len(img.shape) == 2:
            img_rgb = np.stack([img]*3, axis=-1)
            img_rgb = np.expand_dims(img_rgb, axis=0)
            return loaded_model.predict(img_rgb)
        return loaded_model.predict(img)

    # Generate explanation
    explanation = explainer.explain_instance(test_img[0].astype('double'), model_predict, top_labels=5, hide_color=0, num_samples=1000)

    # Fetch explanation for the top class
    top_label = explanation.top_labels[0]
    temp, mask = explanation.get_image_and_mask(top_label, positive_only=True, num_features=5, hide_rest=False)

    # Display the explanation
    # plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
    # plt.show()

    # Create the image with boundaries
    image_with_boundaries = mark_boundaries(temp / 2 + 0.5, mask)

    # Convert the image from [0,1] to [0,255] and to uint8
    image_with_boundaries = (image_with_boundaries * 255).astype('uint8')

    # Save the image
    cv2.imwrite(f'lime_outputs/lime_1_{image_path.split("/")[-1]}', cv2.cvtColor(image_with_boundaries, cv2.COLOR_RGB2BGR))

################################################ Text Generation ################################################ 

    # Get textual explanation
    lime_output = {}
    print(f"Explanation for label {top_label}:")
    for area in explanation.local_exp[top_label]:
        print(f"Area: {area[0]}, Importance: {area[1]}")
        lime_output[area[0]] = area[1]


    # Function to generate text explanation
    def generate_text_explanation(lime_output, top_label):
        positive_areas = []
        negative_areas = []
        
        for area, importance in lime_output.items():
            if importance > 0:
                positive_areas.append((area, importance))
            else:
                negative_areas.append((area, importance))
        
        positive_areas = sorted(positive_areas, key=lambda x: x[1], reverse=True)
        negative_areas = sorted(negative_areas, key=lambda x: x[1])
        
        explanation = f"The model predicts that this X-ray most likely falls under label '{label_mapping[top_label]}'. "
        
        if positive_areas:
            explanation += f"It makes this decision primarily based on evidence from areas {[area for area, _ in positive_areas[:3]]} which positively contribute to this prediction. "
        
        if negative_areas:
            explanation += f"However, areas {[area for area, _ in negative_areas[:3]]} provide evidence against this prediction. "
        
        explanation += "Overall, the model's prediction is influenced by these key areas."
        
        return explanation

    # Generate the text explanation
    print("LIME Output:", lime_output)
    print("Top Label:", top_label)

    text_explanation = generate_text_explanation(lime_output, top_label)
    print("Generated Text Explanation:", text_explanation)

################################################ Influential Area Marking ################################################ 

    top_label = explanation.top_labels[0]
    temp, mask = explanation.get_image_and_mask(top_label, positive_only=False, num_features=5, hide_rest=False)

    # Create masks for the most and least influential areas
    most_influential_mask = np.isin(mask, [8, 7, 17])
    least_influential_mask = np.isin(mask, [5, 4, 0])

    # Highlight the most influential areas in green and the least influential areas in red
    highlighted_img = temp / 2 + 0.5  # Convert the image back to the [0, 1] range
    highlighted_img[most_influential_mask] = [0, 1, 0]  # Green for most influential
    highlighted_img[least_influential_mask] = [1, 0, 0]  # Red for least influential

    # Display the highlighted image
    # plt.imshow(highlighted_img)
    # plt.title("Green areas are most influential, Red areas are least influential")
    # plt.axis('off')
    # plt.show()

    # Convert the image from [0,1] to [0,255] and to uint8
    highlighted_img_uint8 = (highlighted_img * 255).astype('uint8')

    # Save the image using OpenCV (note the conversion from RGB to BGR)
    cv2.imwrite(f'lime_outputs/lime_2_{image_path.split("/")[-1]}', cv2.cvtColor(highlighted_img_uint8, cv2.COLOR_RGB2BGR))


################################################ Segment Marking ################################################ 
    segments = explanation.segments

    # Fetch explanation for the top class
    top_label = explanation.top_labels[0]
    lime_output = {}
    for area in explanation.local_exp[top_label]:
        lime_output[area[0]] = area[1]

    # Sort areas by importance and keep only positive ones
    positive_areas = [area for area, importance in sorted(lime_output.items(), key=lambda x: x[1], reverse=True) if importance > 0]

    # Create an image to display
    highlighted_img = temp / 2 + 0.5  # Convert the image back to the [0, 1] range

    # Create a figure and axes
    fig, ax = plt.subplots()

    # Loop through each unique segment index
    for segment_index in positive_areas:
        # Create a mask for the current segment index
        segment_mask = (segments == segment_index)
        
        # Find the centroid of this segment
        coords = np.column_stack(np.where(segment_mask))
        if coords.size == 0:
            continue
        centroid = coords.mean(axis=0)
        
        # Annotate the image with the segment index
        ax.text(centroid[1], centroid[0], str(segment_index), color='white', fontsize=12, ha='center', va='center')

    # Display the image and annotations on the axes
    ax.imshow(highlighted_img)
    ax.axis('off')

    # Save the image using matplotlib's savefig
    save_path = f'lime_outputs/lime_3_{image_path.split("/")[-1]}'
    fig.savefig(save_path, bbox_inches='tight', pad_inches=0)

    # Close the figure to free up memory
    plt.close(fig)