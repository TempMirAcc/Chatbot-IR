from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the pre-trained ResNet50 model
model = ResNet50(weights='imagenet')

def recognize_image(img_path):
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # Predict the content of the image
    preds = model.predict(x)
    return decode_predictions(preds, top=3)[0]

# Test the function
if __name__ == "__main__":
    image_path = "test_image.jpg"  # Replace with your image file path
    predictions = recognize_image(image_path)
    print("Top predictions for the image:")
    for pred in predictions:
        print(f"{pred[1]}: {pred[2]*100:.2f}%")
