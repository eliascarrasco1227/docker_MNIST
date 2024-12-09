# Imagen base con Python
FROM python:3.8-slim

# Directorio de trabajo
WORKDIR /app

# Instalamos Gradio y Requests
RUN pip install gradio requests

# Copiar los archivos del proyecto
COPY . /app

# Exponemos el puerto 7860 para la interfaz gráfica
EXPOSE 7860

# Comando para iniciar Gradio
CMD ["python", "-u", "interface.py"]
