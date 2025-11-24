import streamlit as st
import cv2
import numpy as np
from PIL import Image

# 1. Configuración de la Interfaz
st.set_page_config(page_title="Detector de Asincronías", page_icon="🫁")

st.title("🫁 Monitor de Asincronías")
st.write("""
**Modo Educativo:** Utiliza la cámara para analizar la pantalla del ventilador mecánico.
Asegúrate de que las curvas de Presión y Flujo sean visibles.
""")

# 2. Adquisición de Imagen (Cámara del Celular)
img_file_buffer = st.camera_input("Toma una foto de la pantalla del ventilador")

# 3. Procesamiento Inicial
if img_file_buffer is not None:
    # Convertir los bytes de la imagen a un array de NumPy que OpenCV pueda entender
    bytes_data = img_file_buffer.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    
    # Diagnóstico técnico de la imagen
    alto, ancho, canales = cv2_img.shape
    st.success(f"Imagen capturada exitosamente. Resolución: {ancho}x{alto}px")
    
    # Mostrar la imagen que "ve" el algoritmo (Grayscale para procesamiento)
    gray_image = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
    st.image(gray_image, caption="Vista del Algoritmo (Escala de Grises)", width=300)
    
    st.info("✅ El sistema de visión está listo para recibir los algoritmos de detección.")
