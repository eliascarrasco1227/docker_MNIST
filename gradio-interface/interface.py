import gradio as gr
import requests
import numpy as np
from PIL import Image
import io

def predict_image(image):
    # Convertimos el array de la imagen desde el campo 'composite'
    img_array = np.array(image['composite']).astype('uint8')
    
    # ERROR: COGIAMOS RGB CUANDO EL RESULTADO ESTABA EN ALFA
    # Verificamos si la imagen tiene 4 canales y cogemos solo el alfa
    if img_array.shape[2] == 4:  # Si la imagen tiene 4 canales (RGBA)
        img_array = img_array[...,3]  # Tomamos solo el 4 canal (A)
    
    # Convertimos el array a una imagen (esto ya es un array de 280x280)
    img = Image.fromarray(img_array)

    # Redimensionamos la imagen a 28x28 píxeles
    img = img.resize((28, 28))

    # Guardamos la imagen en un buffer de memoria en formato PNG
    img_buffer = io.BytesIO()
    img.save(img_buffer, format='PNG')
    img_buffer.seek(0)  # Movemos el puntero al inicio del buffer

    # Enviamos la imagen al servidor Flask
    img_data = {'file': ('input_image.png', img_buffer, 'image/png')}
    
    url = 'http://contenedor_inferencia:5000/predict'
    response = requests.post(url, files=img_data)

    # Imprimir el contenido de la respuesta para depurar
    #print(response.content)

    try:
        response_data = response.json()
        if 'prediction' in response_data:
            return response_data['prediction']
        else:
            return f"Error: {response_data}"
    except Exception as e:
        return f"Error al procesar la respuesta: {e}"


# Crear la interfaz con un Sketchpad de 280x280 píxeles
iface = gr.Interface(
    fn=predict_image, 
    inputs=gr.Sketchpad(height=280, width=280),  # Canvas más grande (280x280)
    outputs="text", 
    live=True,
    title="Reconocimiento de dígitos con MNIST mediate Docker",  # Título de la aplicación
    description="Dibuja un dígito del 0 al 9 en el lienzo para predecir el número con IA.",  # Descripción
    theme="compact"  # Opción de tema
)

iface.launch(server_name="0.0.0.0", server_port=7860)

