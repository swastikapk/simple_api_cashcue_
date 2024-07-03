from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array

app = Flask(__name__)
CORS(app, origin='*')

# Load the models
model1 = tf.keras.models.load_model("predksi.h5")
model2 = tf.keras.models.load_model("bill.h5")



img_size = (150, 150)

def prepare_image(file, target_size):
    img = Image.open(file).convert('RGB')
    img = img.resize(target_size)
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img /= 255.0
    return img

@app.route('/predict_model1', methods=['POST'])
@cross_origin()
def predict_model1():
    data = request.json
    Saldo = float(data['Saldo'])
    Kredit = float(data['Kredit'])
    Sisa_Saldo = float(data['Sisa_Saldo'])
    
    # Prepare input data
    input_data = np.array([[Saldo, Kredit, Sisa_Saldo]], dtype=np.float32)
    
    # Predict using the model
    prediction = model1.predict(input_data)
    
    # Get the index of the maximum value in the prediction
    predicted_class = np.argmax(prediction, axis=1)[0]
    
    # Mapping predicted class index to label
    label_map = {0: 'Hemat', 1: 'Normal', 2: 'Boros'}
    predicted_label = label_map[predicted_class]
    
    # Add ranges to response
    response = {
        'prediction': predicted_label,
        'predicted_number':float(predicted_class)
    }
    
    return jsonify(response)

@app.route('/predict_model2', methods=['POST'])
def predict_model2():
    file = request.files['file']
    img = prepare_image(file, target_size=img_size)
    prediction = model2.predict(img)
    label = 'Bon' if prediction[0][0] < 0.5 else 'Bukan Bon'
    return jsonify({'prediction': label})

if __name__ == '__main__':
    app.run(debug=True)
