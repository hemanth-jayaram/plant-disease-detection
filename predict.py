import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

def load_image(img_path, show=False):
    """
    Load and preprocess an image for prediction.
    
    Args:
        img_path: Path to the image file
        show: Boolean, whether to display the image
        
    Returns:
        Preprocessed image tensor
    """
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        img_tensor = image.img_to_array(img)
        img_tensor = np.expand_dims(img_tensor, axis=0)
        img_tensor /= 255.

        if show:
            plt.imshow(img_tensor[0])
            plt.axis('off')
            plt.show()

        return img_tensor
    except Exception as e:
        print(f"Error loading image: {str(e)}")
        print(f"Please check if the image exists at: {img_path}")
        return None

def predict_image(img_path):
    """
    Predict whether an image shows a healthy or diseased plant leaf.
    
    Args:
        img_path: Path to the image file
    """
    # Clean up the path if it has quotes
    img_path = img_path.strip().strip('"').strip("'")

    print(f"Looking for image at: {img_path}")
    print(f"File exists: {os.path.exists(img_path)}")

    if not os.path.exists(img_path):
        print(f"Error: Image file not found at {img_path}")
        print("Please verify the file path and try again.")
        return

    # Load the trained model
    model_path = os.path.join(os.path.dirname(__file__), 'plant_disease_detector.h5')
    print(f"Loading model from: {model_path}")
    
    try:
        model = load_model(model_path)
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        print(f"Please make sure the model file exists at: {model_path}")
        return

    # Load and preprocess the image
    new_image = load_image(img_path, show=True)
    if new_image is None:
        print("Error: Could not load image. Prediction aborted.")
        return

    # Make prediction
    try:
        pred = model.predict(new_image)
        predicted_class = "Healthy" if pred < 0.5 else "Diseased"
        print(f'\nPrediction: {predicted_class}')
        print(f'Confidence: {pred[0][0]:.2%}')
    except Exception as e:
        print(f"Error making prediction: {str(e)}")

if __name__ == "__main__":
    # Use r"..." to avoid issues with backslashes
    test_image_path = r"C:\Users\heman\Plant disease detection Dataset\crab-apple-fruit-tree-rust-d0c0b053-01cd80cdad534ca18eb4c9e0a9a0b247.jpg"
    predict_image(test_image_path)
