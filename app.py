# Ventilator Lab AI - Edición Ultra-Resiliente (8b Optimized)
import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
import matplotlib
from PIL import Image
import io
import google.generativeai as genai
import time
from datetime import datetime

# Configuración Base
matplotlib.use('Agg')
plt.style.use('dark_background')
st.set_page_config(page_title="Ventilator Lab | AI 8b", page_icon="🫁", layout="wide")

# --- Lógica de Caché para evitar re-llamadas innecesarias ---
@st.cache_data(show_spinner=False)
def procesar_ia_con_cache(prompt, _img, _model_name, api_key):
    """Guarda la respuesta en memoria para no gastar cuota si la imagen no cambia."""
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(_model_name)
    try:
        response = model.generate_content([prompt, _img], 
                                         generation_config={"temperature": 0.1, "max_output_tokens": 250})
        return response.text
    except Exception as e:
        return str(e)

# ==========================================
# 1. MOTOR MATEMÁTICO (ULTRA-RÁPIDO)
# ==========================================
def analizar_matematico_lite(signal, tipo):
    if len(signal) < 10: return "Señal insuficiente"
    sig = np.array(signal, dtype=float)
    sig_norm = (sig - np.min(sig)) / (np.max(sig) - np.min(sig) + 1e-6)
    peaks, _ = find_peaks(sig_norm, prominence=0.2, distance=15)
    fr = round((len(peaks) / 10) * 60, 1) if len(peaks) > 0 else 0
    return f"Picos:{len(peaks)} | FR:{fr}rpm | Modo:{tipo}"

# ==========================================
# 2. INTERFAZ Y LÓGICA DE CONTROL
# ==========================================
def main():
    st.title("🫁 VENTILATOR LAB AI (V8b)")
    st.caption("Optimizada para alta disponibilidad y bajo consumo de cuota.")

    # Gestión de API Key (Prioridad a Secrets)
    api_key = st.secrets.get("GOOGLE_API_KEY") or st.sidebar.text_input("🔑 API Key", type="password")

    col_1, col_2 = st.columns([1, 2])

    with col_1:
        st.subheader("⚙️ Parámetros")
        canal = st.selectbox("Canal", ["Presión (Paw)", "Flujo (Flow)"])
        modo = st.selectbox("Modo Vent.", ["VCV", "PCV", "PSV", "Otros"])
        peep = st.number_input("PEEP", 0, 20, 5)
        
    with col_2:
        img_file = st.file_uploader("Subir captura del monitor", type=['jpg', 'png'])
        
        if img_file and api_key:
            # 1. Procesar Imagen
            bytes_data = img_file.getvalue()
            img_pil = Image.open(io.BytesIO(bytes_data))
            
            # Reducción drástica de resolución para ahorrar tokens (800px -> 512px)
            img_pil.thumbnail((512, 512))
            
            # 2. Extracción de señal (Muestreo rápido)
            img_cv = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, np.array([20, 80, 80]), np.array([110, 255, 255]))
            
            signal = []
            h, w = mask.shape
            for c in range(0, w, 4): # Saltamos de a 4 píxeles para velocidad
                col_data = mask[:, c]
                if np.max(col_data) > 0: signal.append(h - np.argmax(col_data))
            
            if signal:
                st.line_chart(signal)
                resumen_mat = analizar_matematico_lite(signal, canal)
                st.info(f"📊 {resumen_mat}")

                if st.button("🧠 EJECUTAR AUDITORÍA IA (8b Mode)"):
                    # Prompt comprimido (Formato Telegrama)
                    prompt_lite = f"Expert: ICU. Mode: {modo}, PEEP: {peep}. Data: {resumen_mat}. Task: Identify asynchronies (DD, HF) in image. Brief response."
                    
                    # Intentar con el modelo 8b (el más resistente)
                    with st.spinner("Consultando modelo de alta disponibilidad..."):
                        resultado = procesar_ia_con_cache(prompt_lite, img_pil, "gemini-1.5-flash-8b", api_key)
                        
                        if "429" in resultado:
                            st.error("⚠️ Cuota global excedida. Google está limitando las peticiones gratuitas. Reintenta en 60s.")
                        else:
                            st.markdown("### 📝 Reporte de Auditoría")
                            st.markdown(resultado)

if __name__ == "__main__":
    main()
