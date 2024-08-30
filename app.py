from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os
from image_recognition import recognize_image
from chatbot import generate_response

app = Flask(__name__)

# Ensure the 'uploads' folder exists
os.makedirs('uploads', exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image_file = request.files['image']
    filename = secure_filename(image_file.filename)
    filepath = os.path.join('uploads', filename)
    image_file.save(filepath)

    # Recognize the image content
    recognized_objects = recognize_image(filepath)
    image_description = ", ".join([obj[1] for obj in recognized_objects])

    # Generate chatbot response
    chatbot_response = generate_response(image_description)

    return jsonify({"response": chatbot_response, "description": image_description})

if __name__ == '__main__':
    app.run(debug=True)
