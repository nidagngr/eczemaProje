from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
import cv2
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
model = load_model("models/model3_mobilenetv2.h5")
img_size = 128
categories = ['eczema', 'normal']

def predict_image(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (img_size, img_size))
    img = img.astype("float32")
    img = preprocess_input(img)  # <--- eğitimle aynı ön işleme burada
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    pred_class = np.argmax(prediction)
    confidence = prediction[0][pred_class]
    return categories[pred_class], confidence


@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    confidence = None
    file_url = None

    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            result, confidence = predict_image(filepath)
            file_url = filepath

    return render_template('index.html', result=result, confidence=confidence, image_url=file_url)

if __name__ == '__main__':
    app.run(debug=True)
