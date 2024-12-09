import os
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import io

# Cargamos el modelo preentrenado de MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(256, activation='relu'),  # Aumentar neuronas
    tf.keras.layers.Dense(128, activation='relu'),  # Otra capa oculta
    tf.keras.layers.Dropout(0.1),  # Dropout layer
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10, validation_split=0.3)

app = Flask(__name__)

def preprocess_image(image_data):
    try:
        # Abrimos la imagen desde los bytes recibidos
        img = Image.open(io.BytesIO(image_data))

        # Convertimos a escala de grises y ajustamos el tamaño a 28x28
        img = img.convert('L')
        img = img.resize((28, 28))

        # Convertimos la imagen en un array de numpy
        img_array = np.array(img)

        # Verificamos si la imagen es completamente blanca
        if np.all(img_array == 255):
            print("La imagen es completamente blanca, no se detectó ningún dibujo.")
            return None  # Imagen vacía

        # ERROR: NO HAY QUE NORMALIZAR
        # Normalizamos la imagen (valores entre 0 y 1)
        #img_array = img_array / 255.0
        #print("Contenido del array de la imagen NORMALIZADA:", img_array)

        # ERROR: NO HAY QUE INVERTIR
        # Invertimos los valores de los píxeles si están invertidos
        #if np.mean(img_array) > 0.9:  # Si la imagen está demasiado clara
         #   img_array = 1.0 - img_array
            #print("Invertimos la imagen (Post-Inversión):", img_array)

        # Añadimos la dimensión de batch
        img_array = np.expand_dims(img_array, axis=0)
        #print("IMAGEN FINAL:", img_array)

        return img_array

    except Exception as e:
        print(f"Error al procesar la imagen: {e}")
        return None

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Recibimos el archivo de la imagen
        file = request.files['file']
        if not file:
            return jsonify({'error': 'No file uploaded'})

        # Leemos los datos de la imagen
        image_data = file.read()
        #print("IMAGEN TOMADA (Bytes Crudos):", image_data[:100])  # Mostramos solo los primeros 100 bytes

        # Procesamos la imagen
        img_array = preprocess_image(image_data)
        if img_array is None:
            print("Imagen vacía o error al procesar la imagen")
            return jsonify({'error': 'Imagen vacía o procesamiento fallido'})

        # Realizamos la predicción
        predictions = model.predict(img_array)
        predicted_label = np.argmax(predictions)
        print(f"Predicción realizada: {predicted_label}")

        return jsonify({'prediction': int(predicted_label)})
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

