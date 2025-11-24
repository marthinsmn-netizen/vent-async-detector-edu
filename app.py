import streamlit as st
import cv2
import numpy as np
from PIL import Image

# --- Configuración de la Página ---
st.set_page_config(page_title="Ventilador Lab", page_icon="🫁")

st.title("🫁 Detector de Asincronías")
st.markdown("""
**Instrucciones:**
1. Acerca la cámara a la pantalla del ventilador.
2. Intenta que las curvas de Presión y Flujo se vean claras.
3. Toma la foto.
""")

# --- Entrada de Cámara ---
# Nota: En móviles, esto abrirá la cámara. En PC, la webcam.
imagen_camara = st.camera_input("Capturar Pantalla del Ventilador")

if imagen_camara is not None:
    # 1. Convertir la imagen de formato web a formato OpenCV
    file_bytes = np.asarray(bytearray(imagen_camara.read()), dtype=np.uint8)
    img_cv = cv2.imdecode(file_bytes, 1)
    
    # 2. Pre-procesamiento (Lo que "ve" el algoritmo)
    # Convertimos a escala de grises (elimina ruido de colores)
    img_gris = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    
    # 3. Detección de Bordes (Algoritmo Canny)
    # Esto nos muestra si somos capaces de separar las curvas del fondo
    bordes = cv2.Canny(img_gris, 100, 200)

    # --- Visualización de Resultados ---
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(img_camara, caption="Imagen Original")
    
    with col2:
        st.image(bordes, caption="Visión del Algoritmo (Bordes)", clamp=True)

    st.success("✅ Procesamiento de imagen exitoso. El sistema está listo para análisis avanzado.")
    
    # Datos técnicos para el alumno (Feedback educativo)
    h, w, c = img_cv.shape
    st.caption(f"Resolución analizada: {w}x{h} píxeles")
