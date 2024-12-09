# Imagen base con TensorFlow
FROM tensorflow/tensorflow:2.8.0

# Directorio de trabajo
WORKDIR /app

# Instalamos Flask y Pillow para manejar imágenes
RUN pip install flask Pillow

# Copiamos los archivos del proyecto al contenedor
COPY . /app

# Exponemos el puerto 5000 para la API de inferencia
EXPOSE 5000

# Comando para iniciar el servidor Flask
CMD ["python", "-u", "inference.py"]
