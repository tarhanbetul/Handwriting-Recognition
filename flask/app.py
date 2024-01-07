# from flask import Flask, request, jsonify
# import os
# from werkzeug.utils import secure_filename
#
# app = Flask(__name__)
#
# UPLOAD_FOLDER = 'uploads'
# ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
#
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# print ("ttttttt")
# def allowed_file(filename):
#     return '.' in filename and \
#            filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
#
# @app.route('/upload', methods=['POST'])
# def upload_file():
#     print("333ggg")
#     if 'file' not in request.files:
#         return jsonify({'error': 'No file part'})
#
#     file = request.files['file']
#     print("fffffff",file)
#     if file.filename == '':
#         return jsonify({'error': 'No selected file'})
#
#     if file and allowed_file(file.filename):
#         print("fddddddffffff", file)
#         # Windows'ta dosya yollarını düzenlemek için os.path.join kullanın
#         filename = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
#         print("bbbbbbbb", filename)
#         file.save(filename)
#         print("ccccc", file)
#         return jsonify({'success': True, 'message': 'File uploaded successfully'})
#
#     return jsonify({'error': 'Invalid file format'})
#
# if __name__ == '__main__':
#     app.run(debug=True)

# from flask import Flask, request, jsonify
# import os
# from werkzeug.utils import secure_filename
#
# import pytesseract
# from PIL import Image
# from flask_cors import CORS
# import torch
# import torchvision.models as models
#
# # MNIST veri seti üzerinde eğitilmiş bir modeli yükle
# model = models.resnet18(pretrained=True)
#
# pytesseract.pytesseract.tesseract_cmd = r'C:\Users\numan\AppData\Local\Programs\Python\Python311\Scripts'
#
#
# app = Flask(__name__)
# CORS(app)
#
# UPLOAD_FOLDER = 'uploads'
# ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
#
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
#
# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
#
# @app.route('/upload', methods=['POST'])
# def upload_file():
#     print ("yyyhfgbf")
#     if 'file' not in request.files:
#         return jsonify({'error': 'No file part'})
#
#     file = request.files['file']
#
#     if file.filename == '':
#         return jsonify({'error': 'No selected file'})
#
#     if file and allowed_file(file.filename):
#         print ("seksekskeksekskeseee")
#         filename = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
#         file.save(filename)
#
#         # OCR işlemi
#         image = Image.open(filename)
#         print("-----------------------------------------------------6666----------------------------------------------------------")
#         text = pytesseract.image_to_string(image, lang='eng')
#
#         print("OCR Resulttttttttttttttt:", text)
#
#         return jsonify({'success': True, 'message': 'File uploaded successfully', 'OCR_result': text})
#
#     return jsonify({'error': 'Invalid file format'})
#
# if __name__ == '__main__':
#     app.run(debug=True)
#
# from flask import Flask, request, jsonify
# import torch
# from torchvision import transforms
# from PIL import Image
# from flask_cors import CORS
#
# app = Flask(__name__)
# CORS(app)
# # Modeli tanımla ve eğitilmiş ağırlıkları yükle
# model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
# print ("burdamiyiz")
# #model = models.resnet18(pretrained=True)
# model.eval()
#
# # Ön işleme için transform tanımla
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.Grayscale(num_output_channels=3),  # Model RGB görüntü bekliyor
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])
#
# print ("burdamiyiz777777")
# # Flask endpoint'i
# @app.route('/predict', methods=['POST'])
# def predict():
#     print ("----------00000000---------")
#     # Kullanıcının yüklediği resmi al
#     uploaded_file = request.files['file']
#
#     if uploaded_file is None:
#         return jsonify({'error': 'No file uploaded'})
#
#     # Resmi ön işle
#     img = Image.open(uploaded_file).convert('RGB')
#     img = transform(img)
#     img = img.unsqueeze(0)  # Batch boyutunu ekleyin
#
#     # Model ile tahmin yap
#     with torch.no_grad():
#         output = model(img)
#         print ("outputttttt",output)
#     # Tahmin sonuçlarını işle
#     _, predicted_class = torch.max(output, 1)
#     predicted_class = predicted_class.item()
#     print("predicted_classpredicted_classpredicted_class", predicted_class)
#     return jsonify({'predicted_class': predicted_class})
#
#
# if __name__ == '__main__':
#     print ("birede")
#     app.run(debug=True)
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
import base64
from werkzeug.datastructures import FileStorage
from collections import Counter

app = Flask(__name__)
#CORS(app, origins="http://localhost:3000")
CORS(app, resources={r"/predict/*": {"origins": "*"}})
# Önceden eğitilmiş MobileNetV2 modelini yükle
model = MobileNetV2(weights='imagenet')

# Görüntüyü ön işleme fonksiyonu
def preprocess_image(base64_content):
    if isinstance(base64_content, FileStorage):
        # FileStorage objesini base64 ile kodlu string'e dönüştür
        base64_content = base64.b64encode(base64_content.read()).decode('utf-8')

    print("yesssss")
    img_data = base64.b64decode(base64_content)
    img_path = "temp.jpg"
    with open(img_path, 'wb') as f:
        f.write(img_data)

    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# Flask endpoint'i
@app.route('/predict', methods=['POST'])
def predict():
    print("1.Adım",request.files)
    if request.method == 'POST':
        if len(request.files) == 0:
            print("files yok")
        #
        #     if uploaded_file is None:
        # print("request.data['file']",request.data.get('file')
        print ("Content-Type",request)
        file_content = request.files['file']  # request.data['file']

        if file_content is None:
            return jsonify({'error': 'No file content provided'})
        print("2.Adım")

        print("3..Adım",file_content)
        # Resmi ön işle
        img_array = preprocess_image(file_content)

        # Model ile tahmin yap
        predictions = model.predict(img_array)
        decoded_predictions = decode_predictions(predictions, top=3)[0]

        # Tahmin sonuçlarını işle
        results = [{'label': label, 'probability': float(prob)} for (imagenet_id, label, prob) in decoded_predictions]
        print ("r3esults",results)
        return jsonify({'predictions': results})
    else:
        print("----------41241234231423423423423------------------------------------numan---------------------------------------------")
        return jsonify({'error': 'Geçersiz istek'})

if __name__ == '__main__':
    app.run(debug=True)

