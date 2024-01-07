from flask import Flask, request, jsonify
from tensorflow import keras
import numpy as np
from PIL import Image
from flask_cors import CORS
from collections import Counter
from collections import defaultdict

app = Flask(__name__)
CORS(app, resources={r"/predict/*": {"origins": "*"}})

# TensorFlow modelini yükle
model = keras.models.load_model('trainedmodel/mnist-fashion-model.h5')

def preprocess_image(img):
    img = img.convert('L')  # image grey
    img = img.resize((28, 28))  # image 28x28 piksel
    img_array = np.array(img)  # NumPy dizisine dönüştür
    img_array = img_array / 255.0  # 0-1 aralığında ölçeklendirelim
    img_array = np.expand_dims(img_array, axis=0)  # Batch boyutunu ekleyerek 4D tensöre çevirelim
    return img_array

def find_most_common_label(predictions):
    # Labelleri and skorları dict
    label_scores = defaultdict(float)

    for prediction in predictions:
        for score in prediction:
            # Score string
            score_str = str(score)
            # Score'da '-' işareti varsa, '-' sonrasındaki kısmı label olarak, önceki kısmı skor olarak al
            if '-' in score_str:
                label, score_part = score_str.split('-')

    most_common_label =  max(score_part[1])

    # Distinct label ve total score one dict
    distinct_labels = {label: label_scores[label] for label in label_scores}

    return {'distinct_labels': distinct_labels, 'most_common_label': most_common_label, 'most_common_score': label_scores[most_common_label]}


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'})

    uploaded_file = request.files['file']

    if uploaded_file.filename == '':
        return jsonify({'error': 'No selected file'})

    try:
        img = Image.open(uploaded_file)
        img_array = preprocess_image(img)
        # Model ile tahmin yapalım
        predictions = model.predict(img_array)

        result = find_most_common_label(predictions)

        return jsonify({'predicted_label': int(result['most_common_label'])})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
