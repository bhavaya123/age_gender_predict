import os
import numpy as np
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from keras.preprocessing import image
from keras.utils import img_to_array
from keras.models import load_model

# Constants and configurations
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'  # Define upload folder
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}  # Allowed image extensions

# Helper function for file type validation
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('intro.html')
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(url_for('index', error="No file selected"))

    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('index', error="No selected file"))

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Preprocess image for model prediction
        img = image.load_img(file_path, target_size=(128, 128), color_mode='grayscale')
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0  # Normalize pixel values

        # Load the pre-trained Keras model (replace with your model path)
        model_path = 'myKeras.keras'
        model = load_model(model_path,compile=False)

        # Make predictions using the model
        predictions = model.predict(img_array)

        # Process outputs (assuming gender and age, adjust indices if needed)
        gender_prob, age_pred = predictions[0][0], predictions[1][0]
        gender_result = "Female" if gender_prob >= 0.5 else "Male"
        age_result = int(age_pred)

        return render_template('result.html', gender=gender_result, age=age_result)

    return redirect(url_for('index', error="Invalid file type"))

if __name__ == '__main__':
    # Create the upload folder if it doesn't exist
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])

    # Run the Flask app in debug mode for development
    app.run(debug=True, host='0.0.0.0', port=5000)
