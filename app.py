from flask import Flask, render_template, request, jsonify
import base64
from PIL import Image
import numpy as np
import io
import tensorflow as tf
import os
from tensorflow.keras.models import load_model


app = Flask(__name__)

ruta_modelo = os.path.join(os.path.expanduser('~'), 'Descargas', 'mnist_model.h5')

# Cargar el modelo
modelo= load_model(ruta_modelo)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recognize', methods=['POST'])
def recognize():
    # Obtener datos de la imagen desde la solicitud POST
    image_data = request.form['imageData'].split(',')[1]
    
    # Decodificar la imagen en formato base64
    img = Image.open(io.BytesIO(base64.b64decode(image_data)))
    
    # Preprocesamiento de la imagen para que coincida con el formato del modelo
    img = img.resize((28, 28)).convert('L')
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape((1, 28, 28, 1))

    # Realizar la predicción
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)

    return jsonify({'predicted_class': int(predicted_class)})

if __name__ == '__main__':
    app.run(debug=True)
